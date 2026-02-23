import torch
import torch.nn as nn
from einops import rearrange
# from ..modules.transformer import BasicTransformerBlock, PosEmbedding
from .CPAttn_utils import get_query_value
from .croco.blocks import Block_cp, DecoderBlock, PatchEmbed


class PosEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        # self.funcs = [torch.sin, torch.cos]
        # self.out_channels = in_channels*(len(self.funcs)*N_freqs)
        if N_freqs <= 80:
            base = 2
        else:
            base = 5000**(1/(N_freqs/2.5))
        if logscale:
            freq_bands = base**torch.linspace(0,
                                              N_freqs-1, N_freqs)[None, None]
        else:
            freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        shape = x.shape[:-1]
        x = x.reshape(-1, 2, 1)

        encodings = x * self.freq_bands
        sin_encodings = torch.sin(encodings)  # (n, c, num_encoding_functions)
        cos_encodings = torch.cos(encodings)
        pe = torch.cat([sin_encodings, cos_encodings], dim=1)
        pe = pe.reshape(*shape, -1)
        return pe

def get_indices(num):
    mapping = {
        0: [1, 2, 3, 4],
        1: [0, 2, 4, 5],
        2: [0, 1, 3, 5],
        3: [0, 2, 4, 5],
        4: [0, 1, 3, 5],
        5: [1, 2, 3, 4]
    }
    return mapping[num]

class CPAttn(nn.Module):
    def __init__(self, dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                norm_layer,
                rope,
                attn_implementation,
                attn_bias_for_inference_enabled, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.rope = rope
        self.transformer = Block_cp(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x, correspondences, img_h, img_w, R, K, m):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        outs = []

        for i in range(m):
            # indexs = [(i-1+m) % m, (i+1) % m]
            indexs = get_indices(i)

            xy_l=correspondences[:, i, indexs]
            xy_r=correspondences[:, indexs, i]
           
            x_left = x[:, i]
            x_right = x[:, indexs]
            
            R_right = R[:, indexs]
            K_right = K[:, indexs]

            l = R_right.shape[1]
            
            R_left = R[:, i:i+1].repeat(1, l, 1, 1)
            K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)
            
            homo_r = (K_left@torch.inverse(R_left) @
                      R_right@torch.inverse(K_right))

            homo_r = rearrange(homo_r, '(b l) h w -> b l h w', b=xy_r.shape[0])
            query, key_value, key_value_xy, mask = get_query_value(
                x_left, x_right, xy_l, homo_r, img_h, img_w)

            key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
            key_value_pe = self.pe(key_value_xy)

            key_value = rearrange(
                key_value, 'b l c h w-> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (key_value + key_value_pe)*mask[..., None]

            query = rearrange(query, 'b c h w->(b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(
                query.shape[0], 1, 2, device=query.device))

            out = self.transformer(query, key_value, query_pe=query_pe, mask=mask)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out

