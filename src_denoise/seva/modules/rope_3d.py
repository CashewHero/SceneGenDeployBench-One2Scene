# 3D RoPE (Rotary Position Embedding) for multi-view transformer models
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast as amp
from typing import Tuple, Optional


def get_sequence_parallel_world_size() -> int:
    """Get the world size for sequence parallel."""
    # 如果没有使用序列并行，返回1
    try:
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1
    except:
        return 1


def get_sequence_parallel_rank() -> int:
    """Get the rank for sequence parallel."""
    # 如果没有使用序列并行，返回0
    try:
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    except:
        return 0


def pad_freqs(original_tensor: Tensor, target_len: int) -> Tensor:
    """Pad frequency tensor to target length."""
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor[:target_len]
    
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp(enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> Tensor:
    """Generate RoPE parameters."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp(enabled=False)
def rope_apply(x: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
    """
    Apply 3D RoPE to input tensor.
    
    Args:
        x: Input tensor [B, L, N, C]
        grid_sizes: Grid sizes [B, 3] containing (frames, height, width)
        freqs: Frequency tensor [M, C // 2]
    
    Returns:
        Output tensor with RoPE applied
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs for 3 dimensions (frame, height, width)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        
        # Create frequency grids for each dimension
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # frame dimension
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # height dimension
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)   # width dimension
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding with sequence parallelism
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding module."""
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 8192, 
        theta: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequency parameters
        self.register_buffer(
            'freqs', 
            rope_params(max_seq_len, dim, theta),
            persistent=False
        )
    
    def forward(self, x: Tensor, grid_sizes: Tensor) -> Tensor:
        """
        Apply 3D RoPE to input tensor.
        
        Args:
            x: Input tensor [B, L, N, C]
            grid_sizes: Grid sizes [B, 3] for (frames, height, width)
        
        Returns:
            Output tensor with RoPE applied
        """
        return rope_apply(x, grid_sizes, self.freqs)


class RoPEMultiViewAttention(nn.Module):
    """Multi-view attention with 3D RoPE support."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
        use_rope: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_rope = use_rope
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), 
            nn.Dropout(dropout)
        )
        
        if use_rope:
            self.rope = RoPE3D(dim_head, max_seq_len, theta)
    
    def forward(
        self, 
        x: Tensor, 
        context: Optional[Tensor] = None, 
        grid_sizes: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with optional 3D RoPE.
        
        Args:
            x: Input tensor [B, L, C]
            context: Context tensor [B, L, C] (optional)
            grid_sizes: Grid sizes [B, 3] for RoPE (optional)
        
        Returns:
            Output tensor [B, L, C]
        """
        b, l, c = x.shape
        ctx = context if context is not None else x
        
        q = self.to_q(x)  # [B, L, inner_dim]
        k = self.to_k(ctx)  # [B, L, inner_dim] 
        v = self.to_v(ctx)  # [B, L, inner_dim]
        
        # Reshape for multi-head attention
        q = q.view(b, l, self.heads, self.dim_head).transpose(1, 2)  # [B, H, L, D]
        k = k.view(b, l, self.heads, self.dim_head).transpose(1, 2)  # [B, H, L, D]
        v = v.view(b, l, self.heads, self.dim_head).transpose(1, 2)  # [B, H, L, D]
        
        # Apply 3D RoPE if enabled and grid_sizes provided
        if self.use_rope and grid_sizes is not None:
            # Reshape for RoPE: [B, L, N, C] format
            q_rope = q.transpose(1, 2).contiguous()  # [B, L, H, D]
            k_rope = k.transpose(1, 2).contiguous()  # [B, L, H, D]
            
            q_rope = self.rope(q_rope, grid_sizes)
            k_rope = self.rope(k_rope, grid_sizes)
            
            q = q_rope.transpose(1, 2)  # [B, H, L, D]
            k = k_rope.transpose(1, 2)  # [B, H, L, D]
        
        # Compute attention scores
        scale = self.dim_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, L, D]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(b, l, -1)  # [B, L, inner_dim]
        
        return self.to_out(out)


def create_3d_grid_sizes(batch_size: int, num_frames: int, height: int, width: int, device: torch.device) -> Tensor:
    """
    Create grid sizes tensor for 3D RoPE.
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames/views
        height: Height of each frame
        width: Width of each frame
        device: Target device
    
    Returns:
        Grid sizes tensor [B, 3]
    """
    return torch.tensor([[num_frames, height, width]], device=device).repeat(batch_size, 1)


# Testing function
if __name__ == "__main__":
    print("Testing 3D RoPE implementation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 256
    num_heads = 8
    dim_head = 64
    num_frames = 16
    height = 4
    width = 4
    
    # Create test data
    x = torch.randn(batch_size, seq_len, num_heads * dim_head, device=device)
    grid_sizes = create_3d_grid_sizes(batch_size, num_frames, height, width, device)
    
    print(f"Input shape: {x.shape}")
    print(f"Grid sizes: {grid_sizes}")
    
    # Test RoPE attention
    rope_attn = RoPEMultiViewAttention(
        query_dim=num_heads * dim_head,
        heads=num_heads,
        dim_head=dim_head,
        use_rope=True
    ).to(device)
    
    # Test without RoPE
    normal_attn = RoPEMultiViewAttention(
        query_dim=num_heads * dim_head,
        heads=num_heads,
        dim_head=dim_head,
        use_rope=False
    ).to(device)
    
    with torch.no_grad():
        rope_output = rope_attn(x, grid_sizes=grid_sizes)
        normal_output = normal_attn(x)
    
    print(f"RoPE output shape: {rope_output.shape}")
    print(f"Normal output shape: {normal_output.shape}")
    print(f"RoPE output mean: {rope_output.mean().item():.6f}")
    print(f"Normal output mean: {normal_output.mean().item():.6f}")
    print(f"Difference: {(rope_output - normal_output).abs().mean().item():.6f}")
    
    print("✓ 3D RoPE test completed successfully!")
