from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .croco.blocks import DecoderBlock, Block
from .croco.croco import CroCoNet
from .croco.croco_fast3r import CroCoNet_fast3r
from .croco.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid

from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding
# from diffusers import AutoencoderKL
import numpy as np
from einops import rearrange
from .CPAttn import CPAttn
from .CPAttn_utils import get_correspondences
import math
import torch.nn.functional as F

inf = float('inf')


def get_position_encoding(pos, d_model):
    # 生成联合频率
    i = torch.arange(d_model, dtype=torch.float32, device=pos.device)
    freq = 1.0 / (10000 ** (2 * i / d_model))
    
    # 计算联合角度
    u = pos[..., 0]
    v = pos[..., 1]
    angles = u.unsqueeze(-1)*freq + v.unsqueeze(-1)*freq  # [6,32,32,d_model]
    
    pe = torch.empty(*pos.shape[:-1], d_model, device=pos.device)
    pe[..., 0::2] = torch.sin(angles[..., :d_model//2])
    pe[..., 1::2] = torch.cos(angles[..., :d_model//2])
    return pe

# class LearnableUVGrid(nn.Module):
#     def __init__(self, grid_size=64, dim=768):
#         super().__init__()
#         self.grid = nn.Parameter(torch.randn(grid_size, grid_size, dim))
    
#     def forward(self, uv):
#         # 输入uv: [B, N, 2] 归一化到[0,1]
#         grid = self.grid.unsqueeze(0)  # [1, grid_size, grid_size, dim]
#         return F.grid_sample(grid, uv * 2 - 1, align_corners=True)
    
class LearnableUVGrid(nn.Module):
    def __init__(self, grid_size=64, dim=768):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(grid_size, grid_size, dim))
    
    def forward(self, uv):
        # 输入uv: [B, H, W, 2] 归一化到[0,1]
        B = uv.shape[0]
        
        # grid需要从[grid_size, grid_size, dim]转换为[B, dim, grid_size, grid_size]
        grid = self.grid.permute(2, 0, 1).unsqueeze(0)  # [1, dim, grid_size, grid_size]
        grid = grid.expand(B, -1, -1, -1)  # [B, dim, grid_size, grid_size]
        
        # 归一化到[-1, 1]
        uv = uv * 2 - 1
        
        # grid_sample输出: [B, dim, H, W]
        output = F.grid_sample(grid, uv, align_corners=True)
        
        # 转换维度顺序为[B, H, W, dim]
        output = output.permute(0, 2, 3, 1)
        
        return output

croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 1024,
        'enc_num_heads': 16,
        'dec_num_heads': 16,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}


@dataclass
class BackboneFast3rCfg:
    name: Literal["fast3r"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt


class BackboneFast3r(CroCoNet_fast3r):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneFast3rCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet_fast3r.__init__)
  
        super().__init__(**croco_params[cfg.model])

        self.uv_embed = LearnableUVGrid(grid_size=64, dim=1024)
        # self.vae = AutoencoderKL.from_pretrained(
        #     "stabilityai/stable-diffusion-2-1-base", subfolder="vae", torch_dtype=torch.float32, use_safetensors=True)
        # self.vae.eval()
        # self.vae.requires_grad_(False)

        # if cfg.asymmetry_decoder:
        #     self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R
    
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(self.dec_embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        # self.dec_blocks = nn.ModuleList([
        #     Block(enc_embed_dim, dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=None)   for i in range(dec_depth)])
        
        self.dec_blocks = nn.ModuleList([
            Block(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                rope=None,
                attn_implementation='flash_attention',
                attn_bias_for_inference_enabled=False
            ) for _ in range(dec_depth)
        ])

        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    # @torch.no_grad()
    # def encode_image_vae(self, x_input, vae):
    #     b = x_input.shape[0]

    #     # x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
    #     # x_input = x_input.reshape(-1,
    #     #                           x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
    #     z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

    #     z = z.sample()
    #     # z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
    #     #               z.shape[-1])  # (bs, 2, 4, 64, 64)

    #     # use the scaling factor from the vae config
    #     z = z * vae.config.scaling_factor
    #     z = z.float()
    #     return z

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0
        image_ids[:, 1] = 1
        image_ids[:, 2] = 2
        image_ids[:, 3] = 3
        image_ids[:, 4] = 4
        image_ids[:, 5] = 5

        # # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        # per_rank_generator = self._generate_per_rank_generator()

        # # Generate random non-repeating IDs for the remaining views using the generator
        # for b in range(batch_size):
        #     # Use the torch.Generator for randomness to ensure randomness between forward passes
        #     random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
        #     image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[0].shape[0] // num_views

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        # image = self.encode_image_vae(image, self.vae)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, intrinsics_embed1=None, intrinsics_embed2=None):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0),
                                             torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0) if intrinsics_embed1 is not None else None)
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, intrinsics_embed1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, intrinsics_embed2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        intrinsics_embed1 = view1.get('intrinsics_embed', None)
        intrinsics_embed2 = view2.get('intrinsics_embed', None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, intrinsics_embed1, intrinsics_embed2)
        else:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f, pos, extra_embed1=None, extra_embed2=None):
        # final_output = [(f1, f2)]  # before projection
        # f = torch.cat([f1, f2], dim=1)

        # pos = torch.cat([pos1, pos2], dim=1)
        # if extra_embed1 is not None:
        #     f1 = torch.cat((f1, extra_embed1), dim=-1)
        # if extra_embed2 is not None:
        #     f2 = torch.cat((f2, extra_embed2), dim=-1)

        # # project to decoder dim
        # f1 = self.decoder_embed(f1)
        # f2 = self.decoder_embed(f2)

        
        final_output = [f]  # before projection

        f = self.decoder_embed(f)

        image_pos = self._get_random_image_pos(encoded_feats=f,
                                                   batch_size=final_output[0].shape[0],
                                                   num_views=6,
                                                   max_image_idx=200,
                                                   device=f.device)
        

        
        # Apply positional embedding based on image IDs and positions
        f += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting
        # pano_pos = self.image_idx_emb[pos]

        for blk in self.dec_blocks:
            f = blk(f, pos)
            final_output.append(f)

        f = self.dec_norm(f)
        final_output[-1] = f

        # final_output.append((f1, f2))
        # for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
        #     # img1 side
        #     f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
        #     # img2 side
        #     f2, _ = blk1(*final_output[-1][::-1], pos2, pos1)
        #     # store the result
        #     final_output.append((f1, f2))

        # # normalize last output
        # del final_output[1]  # duplicate with final_output[0]
        # final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return final_output

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, image_h, image_w = context["image"].shape
        device = context["image"].device

        intrinsics_embed = None

        context_image = rearrange(context["image"], "b v d h w  -> (b v) d h w")
        true_shape = torch.tensor(context_image.shape[-2:])[None].repeat(context_image.shape[0], 1)
        out, pos, _ = self._encode_image(context_image,
                                             true_shape,
                                             torch.cat((intrinsics_embed, intrinsics_embed), dim=0) if intrinsics_embed is not None else None)


        out = rearrange(out, "(b v) hw d -> b (v hw) d", b=b, v=v)
        if self.intrinsics_embed_loc == 'decoder':
            # FIXME: downsample is hardcoded to 16
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree, downsample=16, merge_hw=True)
            dec1, dec2 = self._decoder(out, pos, intrinsic_emb[:, 0], intrinsic_emb[:, 1])
        else:
            dec = self._decoder(out, pos)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec = list(dec)
            for i in range(len(dec)):
                dec[i] = rearrange(dec[i], "b (v hw) d -> (b v) hw d", v=v)

        if return_views:
            return dec, true_shape
        return dec1, dec2
    
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        # for _ in range(self.aa_block_size):
        tokens = self.frame_blocks[frame_idx](tokens, xpos=pos)
        frame_idx += 1
        intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            # tokens = tokens.view(B, S, P, C).view(B, S * P, C)
            tokens =  rearrange(tokens, "(b v) p c -> b (v p) c", b=B, v=S)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        # for _ in range(self.aa_block_size):
        tokens = self.dec_blocks[global_idx](tokens, xpos=pos)
        global_idx += 1
        tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        intermediates.append(tokens)

        return tokens, global_idx, intermediates

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
