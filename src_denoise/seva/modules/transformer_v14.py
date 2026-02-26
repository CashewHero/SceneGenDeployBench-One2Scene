import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import deepspeed
import math
import xformers.ops as xops


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor, source_view_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Handle the Sequential layers manually to pass source_view_mask to LoRA_Linear
        x = self.net[0](x)  # GEGLU
        x = self.net[1](x)  # Dropout
        x = self.net[2](x)  # LoRA_Linear
        return x


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    # flashattn
    # def _perform_attention(self, q, k, v):
    #     q, k, v = map(
    #         lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads).contiguous(),
    #         (q, k, v),
    #     )
    #     try:
    #         with torch.amp.autocast("cuda", dtype=torch.float16): # !!! bfloat16 fail to convergency due to limited precision
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

        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        out = self._perform_attention(q, k, v)
        
        # Manually handle LoRA_Linear within Sequential
        out_linear, out_dropout = self.to_out[0], self.to_out[1]
        out = out_linear(out)
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
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        source_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout)
        self.attn1 = Attention(
            query_dim=inner_dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
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

        ff_in_out = self.ff_in(self.norm_in(x_rearranged)) + x_rearranged
        attn1_out = self.attn1(self.norm1(ff_in_out), context=None) + ff_in_out
        attn2_out = self.attn2(self.norm2(attn1_out), context=context) + attn1_out
        ff_out = self.ff(self.norm3(attn2_out))
        
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [
                TransformerBlockTimeMix(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
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
        x_projected = self.proj_in(x_rearranged )

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

            x_projected_out = self.proj_out(current_x)
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
