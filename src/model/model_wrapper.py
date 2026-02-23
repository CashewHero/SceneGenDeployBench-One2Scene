from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim

from .types import Gaussians
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..Depth_Anything.depth_anything_v2.dpt import DepthAnythingV2
from .c2e import sample_cubefaces, sample_cubefaces_np
from .losse import GradientLoss_Li, Silog_Loss, EPNLoss
from .metric_collector import DepthMetricCollector, compute_depth_metrics
from .storepoint import storePly
import os
from .ply2gs import read_3dgs_ply_to_gaussians
from plyfile import PlyData

ply_path = '/home/pengfei_wang/NoPoSplat/gaussians.ply'

def load_splats_as_gaussians(load_path: str, device: str = "cpu") -> Gaussians:
    """
    从保存的文件中读取splats并转换为Gaussians格式
    """
    # 读取保存的数据
    splats_data = torch.load(load_path, map_location=device)
    
    # 合并球谐函数参数
    # sh0 = splats_data["sh0"]  # shape: (N, 1, 3)
    # shN = splats_data["shN"]  # shape: (N, K, 3)
    # harmonics = torch.cat([sh0, shN], dim=1)  # shape: (N, 1+K, 3)
    # harmonics = harmonics.permute(0, 2, 1)  # 改为 (N, 3, K+1) 格式

    sh0 = splats_data["sh0"]  # (N, 1, 3)
    harmonics = sh0.transpose(1, 2)[None]  # (N, 1, 3) -> (N, 3, 1)


# Gaussians(
#             rearrange(
#                 gaussians.means,
#                 "b v r srf spp xyz -> b (v r srf spp) xyz",
#             ),
#             rearrange(
#                 gaussians.covariances,
#                 "b v r srf spp i j -> b (v r srf spp) i j",
#             ),
#             rearrange(
#                 gaussians.harmonics,
#                 "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
#             ),
#             rearrange(
#                 gaussians.opacities,
#                 "b v r srf spp -> b (v r srf spp)",
#             ),
#         )
    N, total_sh_coeffs, _, _ = harmonics.shape
    full_harmonics = torch.zeros(N, total_sh_coeffs, 3, 25, device=sh0.device, dtype=sh0.dtype)
    
    # 填入SH0（DC分量）
    full_harmonics[:, :, :, 0:1] = harmonics
    # 直接创建Gaussians对象
    gaussians = Gaussians(
        means=splats_data["means"][None],
        covariances=splats_data["covariances"][None],
        harmonics=full_harmonics,
        opacities=splats_data["opacities"][None]
    )
    
    # print(f"Loaded Gaussians: {gaussians.num_gaussians} points on {gaussians.device}")
    return gaussians


def compute_scale_and_shift(prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

def normalize_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    安全归一化到 [0, 1] 范围
    
    Args:
        depth (torch.Tensor): 输入深度图，形状为 (B, H, W)
        eps (float): 极小值，防止除零错误，默认为 1e-6
    
    Returns:
        torch.Tensor: 归一化后的深度图
    """
    depth = depth.float()
    min_val = depth.amin(dim=(1, 2), keepdim=True)  # 逐样本计算最小值
    max_val = depth.amax(dim=(1, 2), keepdim=True)  # 逐样本计算最大值
    
    # 归一化
    normalized = (depth - min_val) / (max_val - min_val + eps)
    return normalized


def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    eval_depth: bool
    eval_nvs: bool

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.depth_loss = Silog_Loss()
        self.GradientLoss = GradientLoss_Li()
        self.EPNLoss = EPNLoss()
        # self.depth_loss = SiLogLoss()
        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # self.DepthAnything_model = DepthAnythingV2(**model_configs[encoder])
        # self.DepthAnything_model.load_state_dict(torch.load(f'/home/pengfei_wang/NoPoSplat/pretrained_weights/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        # self.DepthAnything_model = self.DepthAnything_model.to(DEVICE).eval()
        # freeze_all_params(self.DepthAnything_model)
        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.eval_depth = test_cfg.eval_depth
        self.eval_nvs = test_cfg.eval_nvs
        self.depth_metric_collector = DepthMetricCollector(median_align=False) 


    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = None
        if self.distiller is not None:
            visualization_dump = {}
        gaussians, depth_pred = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)

        # depth_pred = rearrange(depth_pred, "b v d h w -> (b v d) h w", h=h, w=w)
        # depth_pred = rearrange(depth_pred, "b v h w -> (b v) h w")

        
        # depth_pred  = normalize_depth(depth_pred)
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]
        # depth = self.DepthAnything_model.infer_image(target_gt[0,0].permute(1, 2, 0).numpy()*255)

        # with torch.no_grad():
        #     context_img = batch["context"]["image"]
        #     context_img = rearrange(context_img, "b v d h w -> (b v) d h w", h=h, w=w)
        #     context_img = inverse_normalize(context_img)
        #     depth = self.DepthAnything_model.infer_image(context_img.permute(0, 2, 3, 1))
        #     # # depth = normalize_depth(depth)

        #     # target_gt_ = rearrange(target_gt, "b v d h w -> (b v) d h w", h=h, w=w)
        #     # depth_tar = self.DepthAnything_model.infer_image(target_gt_.permute(0, 2, 3, 1))
        #     # depth_tar = normalize_depth(depth_tar)

        # depth_tar_pred = 1/(rearrange(output.depth, "b v h w -> (b v) h w")+1e-6)

        # depth_pred = rearrange(depth_pred, "b v h w d -> b v h w", b=b, v=v)
        pano_depth = sample_cubefaces(depth_pred.squeeze(-1))

        # pano_depth = 1/(pano_depth+1e-6)
        # valid_mask = batch["context"]["depth"] != 0 
        # depth_gt = 1000/(batch["context"]["depth"]+1e-6)

        # pano_depth = normalize_depth(pano_depth)
        pano_depth = torch.clip(pano_depth, 0.01, 110)

        # depth_loss = self.depth_loss(depth_tar_pred, depth.detach()) + self.depth_loss(depth_pred, depth.detach())
        # depth_loss = self.depth_loss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None]) + self.GradientLoss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None]) + self.EPNLoss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None])
        depth_loss = self.depth_loss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None]) + self.GradientLoss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None]) + self.EPNLoss(pano_depth[:,None], batch["context"]["depth"][:,None], batch["context"]["val_mask"][:,None])

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        total_loss = 10*total_loss + depth_loss
        self.log("train/depth_loss", depth_loss)

        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
                                                    visualization_dump['means'][:, 0].squeeze(-2),
                                                    visualization_dump['means'][:, 1].squeeze(-2),
                                                    pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                # f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    # def on_test_epoch_end(self):
    #     print("\n=== Final Test Results ===")
    #     self.depth_metric_collector.compute_metrics()
    #     self.depth_metric_collector.print()

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, depth_pred = self.encoder(
                batch["context"],
                self.global_step,
            )
        # gaussians = load_splats_as_gaussians("/home/pengfei_wang/DreamCube/outputs4/polyhaven_christmas_photo_studio_04_primary/my_gaussians.pth", device="cuda")
        # gaussians = read_3dgs_ply_to_gaussians("/home/pengfei_wang/DreamScene360/output/104/point_cloud/iteration_10000/point_cloud.ply", device="cuda")
        torch.cuda.empty_cache()
        if self.eval_depth:
            depth_pred = depth_pred.squeeze(-1)  # 
            # rearrange(depth_pred, "(b v) h w -> b v h w", b=b, v=v)
            pano_depth = sample_cubefaces(depth_pred)
            # pano_depth = normalize_depth(pano_depth)
            # pano_depth = torch.clip(pano_depth, 0.001, 1.0)*10
            # pano_depth = normalize_depth(pano_depth)
            # pano_depth = torch.clip(pano_depth, 0.01, 1.0)


            # gt_depth = batch["context"]["depth"]
            # mask = batch["context"]["val_mask"] # * batch["context"]["rgb_val_mask"]
            # self.depth_metric_collector.update(gt_depth, pano_depth, mask)
            import torchvision

            # gt = gt_depth
            # pred = pano_depth.unsqueeze(0)
            # mask = mask

            # scale, shift = compute_scale_and_shift(pred, gt, mask)
            # scale = scale.unsqueeze(1).unsqueeze(2)
            # shift = shift.unsqueeze(1).unsqueeze(2)

            # aligned_gt = pano_depth * scale + shift

            
            # torchvision.utils.save_image(vis_depth_map(gt_depth[0]), f'/home/pengfei_wang/NoPoSplat/outputs/s2d3d_gt/{batch["scene"][0][0].split("/")[1]}_{batch["scene"][0][0].split("/")[-1]}')
            # torchvision.utils.save_image(mask.float(), f'/home/pengfei_wang/NoPoSplat/outputs/depth_vggt_s2d3d/{batch["scene"][0][0].split("/")[1]}_mask_{batch["scene"][0][0].split("/")[-1]}')
            # torchvision.utils.save_image(vis_depth_map(pano_depth), f'/home/pengfei_wang/NoPoSplat/outputs/depth_pano360/p_{batch["scene"][0][0].split("/")[1]}_{batch["scene"][0][0].split("/")[-1]}')

            torchvision.utils.save_image(vis_depth_map(pano_depth), f'/home/pengfei_wang/NoPoSplat/outputs/vggt_pano360/p_{batch["scene"][0].split("/")[1]}')

        elif self.eval_nvs:
            
            with self.benchmarker.time("decoder", num_calls=v):
                # output1 = self.decoder.forward(
                #     gaussians,
                #     batch["target"]["extrinsics"][:,:170],
                #     batch["target"]["intrinsics"][:,:170],
                #     batch["target"]["near"][:,:170],
                #     batch["target"]["far"][:,:170],
                #     (h, w),
                # )

                # output2 = self.decoder.forward(
                #     gaussians,
                #     batch["target"]["extrinsics"][:,170:],
                #     batch["target"]["intrinsics"][:,170:],
                #     batch["target"]["near"][:,170:],
                #     batch["target"]["far"][:,170:],
                #     (h, w),
                # )
                # keep_indices = torch.cat([
                #     torch.arange(0, 1),      # 前3张
                #     torch.arange(6, 70)      # 第7张到第70张
                # ])

                output1 = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"][:,:70],
                    batch["target"]["intrinsics"][:,:70],
                    batch["target"]["near"][:,:70],
                    batch["target"]["far"][:,:70],
                    (h, w),
                )


                output2 = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"][:,70:],
                    batch["target"]["intrinsics"][:,70:],
                    batch["target"]["near"][:,70:],
                    batch["target"]["far"][:,70:],
                    (h, w),
                )


            context_img = inverse_normalize(batch["context"]["image"][0])
            # Save images.
            (scene,) = batch["scene"]
            scene = Path(scene)
            save_dir = scene / "render"
            save_dir.mkdir(parents=True, exist_ok=True)



            all_extrinsics = torch.cat([
                batch["context"]["extrinsics"],
                batch["target"]["extrinsics"]
            ], dim=1).squeeze(0)

            seva_c2w = torch.cat([
                batch["context"]["extrinsics"],
                batch["target"]["seva_c2w"]
            ], dim=1).squeeze(0)

            # 连接intrinsics
            all_intrinsics = torch.cat([
                batch["context"]["intrinsics"],
                batch["target"]["intrinsics"]
            ], dim=1).squeeze(0)

            image = torch.cat([
                context_img,
                output1.color[0],
                output2.color[0]
            ], dim=0)

            num_images = image.shape[0]
            conda_mask_1 = torch.zeros(num_images, dtype=torch.float32)
            conda_mask_1[0] = 1

            # 创建conda_mask_6: 前6个为1，其他为0
            conda_mask_6 = torch.zeros(num_images, dtype=torch.float32)
            conda_mask_6[:6] = 1

            def remove_frames(tensor, indices_to_remove=range(1, 6)):
                """去掉指定索引的帧"""
                if tensor is None:
                    return None
                # 创建保留的索引
                total_frames = tensor.shape[1] if tensor.dim() > 1 else tensor.shape[0]
                keep_mask = torch.ones(total_frames, dtype=torch.bool)
                keep_mask[list(indices_to_remove)] = False
                return tensor[:, keep_mask] if tensor.dim() > 1 else tensor[keep_mask]


            save_dict = {
                'cuda_images': image,
                'conda_mask_1': conda_mask_1,
                'conda_mask_6': conda_mask_6,
                'intrinsics': all_intrinsics,
                'extrinsics': all_extrinsics,
                "seva_c2w": seva_c2w,
                'scene': scene
            }
            # 保存到指定路径
            torch.save(save_dict, save_dir / 'data.pth')


            video_path = f'{save_dir}/demo_output_video.mp4'
            fps = 10  # 帧率
            height, width = image.shape[2], image.shape[3]
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            import numpy as np
            # 遍历每一帧，转换为 NumPy 格式并保存到视频
            for i in range(image[6:].shape[0]):  # 遍历帧数
                frame = image[6:][i]  # 取出第 i 帧，形状为 (3, 512, 512)
                frame_np = (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # 转换为 (512, 512, 3)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 的 BGR 格式
                video_writer.write(frame_bgr)  # 写入视频

            # 释放资源
            video_writer.release()
            print(f"Video saved at {video_path}")


            fused_ply_path = os.path.join(save_dir, "fused.ply")
            context_img_flat = context_img.permute(0, 2, 3, 1).reshape(-1, 3)  # [v*h*w, 3]
            fused_rgb = context_img_flat.cpu().numpy() * 255
            fused_xyz = gaussians.means[0].cpu().numpy()
            # 输出稀疏化前后的点云数据
            print(f"Original point cloud shape: {fused_xyz.shape}")
            storePly(fused_ply_path, fused_xyz, fused_rgb)



            


    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                # f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians, depth_pred = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        # depth_pred = 1/(depth_pred+1e-6)
        # depth_pred = rearrange(depth_pred, "b v h w d -> b v d h w", b=b, v=v)
        pano_depth = sample_cubefaces(depth_pred.squeeze(-1))
        pano_depth = vis_depth_map(pano_depth[0])
        pano_depth_gt = vis_depth_map(batch["context"]["depth"][0])
        pano_vis = [pano_depth, pano_depth_gt]

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])


        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        # context = []
        context_rgb = []
        context_depth = []
        for i in range(context_img.shape[0]):
            context_rgb.append(context_img[i])
            context_depth.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context_rgb), "Context"),
            add_label(vcat(*context_depth), "Context Depth"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*pano_vis), "Pano Depth (Prediction)"),
        )
        

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0]
            )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)
        del gaussians, output

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=80)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 60,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians, _ = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
