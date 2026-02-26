import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import deepspeed
import math
import xformers.ops as xops

class LoRA_Linear(nn.Module):
    """
    A linear layer with two independent LoRA adapters, selectable via a mask.
    Preserves original parameter names for compatibility with pretrained weights.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=4, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # Use the same parameter names as nn.Linear for compatibility
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        if rank > 0:
            # LoRA Adapter 1 ('default' for target views)
            self.lora_A1 = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B1 = nn.Parameter(torch.zeros(out_features, rank))

            # LoRA Adapter 2 ('source_lora' for source views)
            self.lora_A2 = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B2 = nn.Parameter(torch.zeros(out_features, rank))
            
            self.scaling = self.alpha / self.rank
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)

    def forward(self, x, source_view_mask=None):
        assert source_view_mask is not None
        # Standard linear transformation
        base_out = F.linear(x, self.weight, self.bias)

        if self.rank == 0 or source_view_mask is None:
            return base_out
            
        # Calculate deltas for both adapters
        delta1 = (F.linear(x, self.lora_A1) @ self.lora_B1.T) * self.scaling
        delta2 = (F.linear(x, self.lora_A2) @ self.lora_B2.T) * self.scaling

        # source_view_mask should be broadcastable to the feature dimension
        final_delta = delta2 * source_view_mask + delta1 * (1 - source_view_mask)

        return base_out + final_delta

    def __repr__(self):
        return f"LoRA_Linear(in={self.in_features}, out={self.out_features}, rank={self.rank})"


class LoRA_Conv2d(nn.Module):
    """
    A Conv2d layer with two independent LoRA adapters, selectable via a mask.
    Preserves original parameter names for compatibility with pretrained weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, rank=4, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use the same parameter names as nn.Conv2d for compatibility
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Store conv parameters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        if rank > 0:
            # LoRA Adapter 1 for target views
            self.lora_A1 = nn.Parameter(torch.zeros(rank, in_channels))
            self.lora_B1 = nn.Parameter(torch.zeros(out_channels, rank))
            
            # LoRA Adapter 2 for source views
            self.lora_A2 = nn.Parameter(torch.zeros(rank, in_channels))
            self.lora_B2 = nn.Parameter(torch.zeros(out_channels, rank))
            
            self.scaling = self.alpha / self.rank
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)

    def forward(self, x, source_view_mask=None):
        assert source_view_mask is not None
        # Standard convolution
        base_out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        if self.rank == 0 or source_view_mask is None:
            return base_out
            
        # Calculate LoRA deltas
        b, c, h, w = x.shape
        
        # Reshape for matrix multiplication: (b, c, h, w) -> (b*h*w, c)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Apply LoRA adapters
        delta1 = (x_reshaped @ self.lora_A1.T) @ self.lora_B1.T  # (b*h*w, out_channels)
        delta2 = (x_reshaped @ self.lora_A2.T) @ self.lora_B2.T
        
        # Reshape back to conv format: (b*h*w, out_channels) -> (b, out_channels, h, w)
        delta1 = delta1.reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.scaling
        delta2 = delta2.reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.scaling
        
        # Apply mask selection
        mask_reshaped = source_view_mask.expand(-1, delta1.shape[1], -1, -1)
        final_delta = delta2 * mask_reshaped + delta1 * (1 - mask_reshaped)
        
        return base_out + final_delta

    def __repr__(self):
        return f"LoRA_Conv2d(in={self.in_channels}, out={self.out_channels}, kernel_size={self.kernel_size[0]}, rank={self.rank})"


def load_pretrained_weights_with_lora(model_with_lora, pretrained_state_dict, strict=True):
    """
    Load pretrained weights into a model with LoRA layers.
    
    Args:
        model_with_lora: Model with LoRA layers
        pretrained_state_dict: State dict from pretrained model
        strict: Whether to strictly enforce that all keys match
    
    Returns:
        Missing keys and unexpected keys
    """
    model_state_dict = model_with_lora.state_dict()
    
    # Filter out LoRA parameters from the current model
    filtered_pretrained = {}
    missing_keys = []
    unexpected_keys = []
    
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_pretrained[key] = value
            else:
                print(f"Shape mismatch for {key}: model {model_state_dict[key].shape} vs pretrained {value.shape}")
        else:
            unexpected_keys.append(key)
    
    # Find missing keys (excluding LoRA parameters)
    for key in model_state_dict.keys():
        if not any(lora_key in key for lora_key in ['lora_A1', 'lora_A2', 'lora_B1', 'lora_B2']):
            if key not in filtered_pretrained:
                missing_keys.append(key)
    
    # Load the filtered state dict
    model_with_lora.load_state_dict(filtered_pretrained, strict=False)
    
    if not strict:
        print(f"Loaded pretrained weights. Missing: {len(missing_keys)} keys, Unexpected: {len(unexpected_keys)} keys")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    return missing_keys, unexpected_keys


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, lora_rank: int = 4):
        super().__init__()
        self.proj = LoRA_Linear(dim_in, dim_out * 2, rank=lora_rank)

    def forward(self, x: torch.Tensor, source_view_mask: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x, source_view_mask).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim, lora_rank=lora_rank), nn.Dropout(dropout), LoRA_Linear(inner_dim, dim_out, rank=lora_rank)
        )

    def forward(self, x: torch.Tensor, source_view_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Handle the Sequential layers manually to pass source_view_mask to LoRA_Linear
        x = self.net[0](x, source_view_mask=source_view_mask)  # GEGLU
        x = self.net[1](x)  # Dropout
        x = self.net[2](x, source_view_mask=source_view_mask)  # LoRA_Linear
        return x


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = LoRA_Linear(query_dim, inner_dim, rank=lora_rank, bias=False)
        self.to_k = LoRA_Linear(context_dim, inner_dim, rank=lora_rank, bias=False)
        self.to_v = LoRA_Linear(context_dim, inner_dim, rank=lora_rank, bias=False)
        
        self.to_out = nn.Sequential(
            LoRA_Linear(inner_dim, query_dim, rank=lora_rank), nn.Dropout(dropout)
        )

    # flashattn
    # def _perform_attention(self, q, k, v):
    #     q, k, v = map(
    #         lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads).contiguous(),
    #         (q, k, v),
    #     )
    #     try:
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16): # !!! bfloat16 fail to convergency due to limited precision
    #             with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    #                 out = F.scaled_dot_product_attention(q, k, v)

    #     except Exception as e:
    #         print(f'Error in _perform_attention: {e}')
    #         raise e

    #     return rearrange(out, "b h l d -> b l (h d)").contiguous()
    
    # xformers
    def _perform_attention(self, q, k, v):
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b l h d", h=self.heads).contiguous(),
            (q, k, v),
        )
        try:
            # xformers
            out = xops.memory_efficient_attention(q, k, v)
            out = rearrange(out, "b l h d -> b l (h d)").contiguous()

            
        except Exception as e:
            print(f'Error in _perform_attention: {e}')
            raise e

        return out

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        source_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        ctx = context if context is not None else x

        q = self.to_q(x, source_view_mask=source_view_mask)
        k = self.to_k(ctx, source_view_mask=source_view_mask)
        v = self.to_v(ctx, source_view_mask=source_view_mask)

        out = self._perform_attention(q, k, v)
        
        # Manually handle LoRA_Linear within Sequential
        out_linear, out_dropout = self.to_out[0], self.to_out[1]
        out = out_linear(out, source_view_mask=source_view_mask)
        out = out_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.ff = FeedForward(dim, dropout=dropout, lora_rank=lora_rank)
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        source_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn1(self.norm1(x), source_view_mask=source_view_mask) + x
        x = self.attn2(self.norm2(x), context=context, source_view_mask=source_view_mask) + x
        x = self.ff(self.norm3(x), source_view_mask=source_view_mask) + x
        return x


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, lora_rank=lora_rank)
        self.attn1 = Attention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, lora_rank=lora_rank)
        self.attn2 = Attention(query_dim=inner_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        source_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, s, _ = x.shape
        x_rearranged = rearrange(x, "(b t) s c -> (b s) t c", t=num_frames).contiguous()
        
        temp_mask = None
        if source_view_mask is not None:
            b = x.shape[0] // num_frames
            # source_view_mask shape is (b*t, 1, 1) or (b*t,), need to reshape properly
            temp_mask = source_view_mask.view(b, num_frames).unsqueeze(-1)  # (b, t, 1)
            temp_mask = temp_mask.unsqueeze(1).repeat(1, s, 1, 1).reshape(-1, num_frames, 1)  # (b*s, t, 1)

        ff_in_out = self.ff_in(self.norm_in(x_rearranged), source_view_mask=temp_mask) + x_rearranged
        attn1_out = self.attn1(self.norm1(ff_in_out), context=None, source_view_mask=temp_mask) + ff_in_out
        attn2_out = self.attn2(self.norm2(attn1_out), context=context, source_view_mask=temp_mask) + attn1_out
        ff_out = self.ff(self.norm3(attn2_out), source_view_mask=temp_mask)
        
        return rearrange(ff_out, "(b s) t c -> (b t) s c", s=s).contiguous()


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        return x_spatial + x_temporal


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = LoRA_Linear(in_channels, inner_dim, rank=lora_rank)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout, lora_rank=lora_rank) for _ in range(depth)]
        )
        self.proj_out = LoRA_Linear(inner_dim, in_channels, rank=lora_rank)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [TransformerBlockTimeMix(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout, lora_rank=lora_rank) for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        source_view_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert context.ndim == 3
        b, c, h, w = x.shape
        x_in = x

        time_context = context
        time_context_first_timestep = time_context[::num_frames]
        time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)

        if self.name in self.unflatten_names:
            context = context[::num_frames]

        x = self.norm(x)
        x_rearranged = rearrange(x, "b c h w -> b (h w) c").contiguous()
        source_view_mask_rearranged = repeat(source_view_mask, "b c h w -> b (h w) c").contiguous() # (b*num_frames 1 1 1) -> (b*num_frames 1 1)
        x_projected = self.proj_in(x_rearranged,source_view_mask_rearranged )

        current_x = x_projected # [(b t) (h w) c]
        current_mask = source_view_mask_rearranged # [(b t) 1 1]
        for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
            if self.name in self.unflatten_names:
                current_x = rearrange(current_x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w).contiguous()
                current_mask = repeat(current_mask, "(b t) 1 1 -> b (t h w) 1", t=num_frames, h=h, w=w).contiguous()

            current_x = torch.utils.checkpoint.checkpoint(block, current_x, context, current_mask, use_reentrant=False)
            
            if self.name in self.unflatten_names:
                current_x = rearrange(current_x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w).contiguous()
                current_mask = rearrange(current_mask, "b (t h w) 1 -> (b t) (h w) 1", t=num_frames, h=h, w=w).mean(1, keepdim=True).contiguous() # (b*num_frames 1 1)

            x_mix = torch.utils.checkpoint.checkpoint(mix_block, current_x, time_context, num_frames, current_mask, use_reentrant=False)
            current_x = self.time_mixer(x_spatial=current_x, x_temporal=x_mix)

            x_projected_out = self.proj_out(current_x, source_view_mask=current_mask)
            x_rearranged_out = rearrange(x_projected_out, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            return x_rearranged_out + x_in


if __name__ == "__main__":
    batch_size = 1
    num_frames = 21
    n_heads = 16
    d_head = 64
    name = "middle_ds8"
    depth = 1
    context_dim = 1024
    height = 32
    width = 32
    dropout = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32
    lora_rank_test = 8
    
    mvtransformer = MultiviewTransformer(
        in_channels=context_dim, 
        n_heads=n_heads, 
        d_head=d_head, 
        name=name, 
        depth=depth, 
        context_dim=context_dim, 
        dropout=dropout,
        lora_rank=lora_rank_test,
    ).to(device, dtype)
    
    print(mvtransformer)

    x = torch.randn(batch_size*num_frames, context_dim, height, width, device=device, dtype=dtype)
    context = torch.randn(batch_size*num_frames, 1, context_dim, device=device, dtype=dtype)
    
    source_view_mask = (torch.rand(batch_size * num_frames, 1,  1, 1, device=device) > 0.5).to(dtype)

    print(f"\nTesting with {int(source_view_mask.sum())} source views and {int((1-source_view_mask).sum())} target views.")

    out = mvtransformer(x, context, num_frames, source_view_mask=source_view_mask)
    print(f"\nOutput shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    print("Manual Dual LoRA test finished successfully!")
