import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
import numpy as np
import os 
import cv2
import torchvision.transforms as transforms
from .utills import e2c
import torch.nn.functional as F
import random

def cassini2Equirec(cassini):
  if cassini.ndim == 2:
    cassini = np.expand_dims(cassini, axis=-1)
    source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3)
  elif cassini.ndim == 3:
    source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3)
  else:
    source_image = cassini

  erp_h = source_image.shape[-1]
  erp_w = source_image.shape[-2]

  theta_erp_start = np.pi - (np.pi / erp_w)
  theta_erp_end = -np.pi
  theta_erp_step = 2 * np.pi / erp_w
  theta_erp_range = np.arange(theta_erp_start, theta_erp_end, -theta_erp_step)
  theta_erp_map = np.array([theta_erp_range for i in range(erp_h)]).astype(np.float32)

  phi_erp_start = 0.5 * np.pi - (0.5 * np.pi / erp_h)
  phi_erp_end = -0.5 * np.pi
  phi_erp_step = np.pi / erp_h
  phi_erp_range = np.arange(phi_erp_start, phi_erp_end, -phi_erp_step)
  phi_erp_map = np.array([phi_erp_range for j in range(erp_w)]).astype(np.float32).T

  theta_cassini_map = np.arctan2(np.tan(phi_erp_map), np.cos(theta_erp_map))
  phi_cassini_map = np.arcsin(np.cos(phi_erp_map) * np.sin(theta_erp_map))

  grid_x = torch.FloatTensor(np.clip(-phi_cassini_map / (0.5 * np.pi), -1, 1)).unsqueeze(-1)
  grid_y = torch.FloatTensor(np.clip(-theta_cassini_map / np.pi, -1, 1)).unsqueeze(-1)
  grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0).repeat_interleave(source_image.shape[0], dim=0)

  sampled_image = F.grid_sample(source_image, grid, mode='bilinear', align_corners=True, padding_mode='border')  # 1, ch, self.output_h, self.output_w

  if cassini.ndim == 3:
    erp = sampled_image.transpose(1, 3).transpose(1, 2).data.numpy()[0].astype(cassini.dtype)
    return erp.squeeze()
  else:
    erp = sampled_image.numpy()
    return erp.squeeze(1)

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list

def get_K_R_tensor(FOV, THETA, PHI, height, width):
    # 计算内参
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], np.float32)
    
    # 归一化内参矩阵到0-1
    K_normalized = K.copy()
    K_normalized[0, 0] /= width
    K_normalized[1, 1] /= height
    K_normalized[0, 2] /= width
    K_normalized[1, 2] /= height
    
    # 计算旋转矩阵
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    
    # 创建W2C矩阵
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = 0
    
    # 转换为tensor
    K_tensor = torch.from_numpy(K_normalized).float()
    W2C_tensor = torch.from_numpy(W2C).float()
    
    return K_tensor, W2C_tensor

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetRE10kCfgWrapper:
    re10k: DatasetRE10kCfg


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetRE10kCfg


@dataclass
class DatasetScannetppCfgWrapper:
    scannetpp: DatasetRE10kCfg


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.is_training = (self.stage == "train")

        if self.stage == "train":
            # self.chunks = np.load(os.path.join(cfg.roots[0], 'train.npy'), allow_pickle=True)
            self.chunks = read_list("/home/pengfei_wang/depthsplat/Structure3D_train.txt")
            # self.chunks = []
            # self.chunks = read_list("/home/pengfei_wang/PanDA/datasets/Structure3D_test.txt")
            # self.chunks_test = read_list("/home/pengfei_wang/PanDA/datasets/matterport3d_train.txt")
            # self.chunks.extend(self.chunks_test)

            self.chunks_test = read_list("/home/pengfei_wang/depthsplat/deep360_train.txt")
            self.chunks.extend(self.chunks_test)
            self.chunks_test = read_list("/home/pengfei_wang/depthsplat/deep360_train.txt")
            self.chunks.extend(self.chunks_test)

            # self.chunks_test = read_list("/home/pengfei_wang/PanDA/datasets/Structure3D_test.txt")
            # self.chunks.extend(self.chunks_test)

        elif self.stage == "val":
            # self.chunks = np.load(os.path.join(cfg.roots[0], 'test.npy'), allow_pickle=True)
            # self.chunks = read_list("/home/pengfei_wang/PanDA/datasets/matterport3d_test.txt")
            # self.chunks = read_list("/home/pengfei_wang/PanDA/datasets/stanford2d3d_test.txt")
            self.chunks = read_list("/home/pengfei_wang/depthsplat/deep360_train.txt")
            self.chunks_test = read_list("/home/pengfei_wang/depthsplat/Structure3D_test.txt")
            self.chunks.extend(self.chunks_test)

        elif self.stage == "test":
            # self.chunks = np.load(os.path.join(cfg.roots[0], 'test.npy'), allow_pickle=True)
            self.chunks = read_list("/home/pengfei_wang/PanDA/datasets/Structure3D_test.txt")
            # self.chunks = read_list("/home/pengfei_wang/PanDA/datasets/deep360_train.txt")

        else:
            raise ValueError(f"Unknown stage: {self.stage}")



        self.e2c = e2c

        # # Collect chunks.
        # self.chunks = []
        # for root in cfg.roots:
        #     root = root / self.data_stage
        #     root_chunks = sorted(
        #         [path for path in root.iterdir() if path.suffix == ".torch"]
        #     )
        #     self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.color_aug= transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            self.color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        transform = transforms.Compose([
            transforms.ToTensor()  # 将图像从 HWC 转换为 CHW 格式，并归一化到 [0, 1]
        ])

        
        
        # 生成相机参数
        u_deg = [0, 90, 180, 270, -90, -90]
        v_deg = [0, 0, 0, 0, 90, -90]
        FOV = 95
        height = width = 512
        self.w = 1024
        self.h = 512
        # 直接生成tensor格式的相机参数
        K_tensors = []
        W2C_tensors = []

        for theta, phi in zip(u_deg, v_deg):
            K, W2C = get_K_R_tensor(FOV, theta, phi, height, width)
            K_tensors.append(K)
            W2C_tensors.append(W2C)

        # 堆叠成批量tensor
        intrinsics = torch.stack(K_tensors)  # [1, 6, 3, 3]
        extrinsics = torch.stack(W2C_tensors)  # [1, 6, 4, 4]


        for chunk_path in self.chunks:
            if '2D_rendering' in chunk_path[0]:
                root_dir = "/data/pengfei_wang/Structured3D"
                scale = 1000.0
                max_depth_meters = 10.0
                min_depth_meters = 0.01

                rgb_name = os.path.join(root_dir, chunk_path[0].lstrip('/'))
                rgb = cv2.imread(rgb_name)
                if rgb is None:
                    print(rgb_name)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                depth_name = os.path.join(root_dir, chunk_path[1])
                gt_depth = cv2.imread(depth_name, -1)
                gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)

                if self.is_training and random.random() > 0.5:
                    rgb = cv2.flip(rgb, 1)
                    gt_depth = cv2.flip(gt_depth, 1)

                cube_depth, _ = self.e2c(np.expand_dims(gt_depth, axis=-1).copy())
                gt_depth = gt_depth.astype(float) / scale
                cube_depth = cube_depth.astype(float) / scale

                gt_depth[gt_depth > max_depth_meters+1] = max_depth_meters + 1
                cube_depth[cube_depth > max_depth_meters+1] = max_depth_meters+1

            else:
                root_dir = "/data/pengfei_wang"
                scale = 1
                max_depth_meters = 100.0
                min_depth_meters = 0.01
                rgb_name = os.path.join(root_dir, chunk_path[0].lstrip('/'))

                # rgb_name = chunk_path[0]
                rgb = cv2.imread(rgb_name)
                if rgb is None:
                    print(rgb_name)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


                # rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
                # depth_name = os.path.join(root_dir, chunk_path[1])
                # gt_depth = cv2.imread(depth_name, -1)

                # if self.is_training and random.random() > 0.5:
                #     rgb = cv2.flip(rgb, 1)
                #     gt_depth = cv2.flip(gt_depth, 1)
                
                # cube_depth, _ = self.e2c(np.expand_dims(gt_depth, axis=-1).copy())
                # gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
                # gt_depth = gt_depth.astype(float) / scale
                # cube_depth = cube_depth.astype(float) / scale
                # gt_depth[gt_depth > max_depth_meters+1] = max_depth_meters + 1
                # cube_depth[cube_depth > max_depth_meters+1] = max_depth_meters + 1


                rgb = cassini2Equirec(rgb)
                depth_name = root_dir + chunk_path[1]
                gt_depth = np.load(depth_name)['arr_0'].astype(np.float32)
                gt_depth = cassini2Equirec(gt_depth)
                gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)

                if self.is_training and random.random() > 0.5:
                    rgb = cv2.flip(rgb, 1)
                    gt_depth = cv2.flip(gt_depth, 1)

                cube_depth, _ = self.e2c(np.expand_dims(gt_depth, axis=-1).copy())
                cube_depth = cube_depth.astype(float) / scale

                gt_depth[gt_depth > max_depth_meters] = max_depth_meters
                cube_depth[cube_depth > max_depth_meters] = max_depth_meters



            # if self.is_training and random.random() > 0.5:
            #     rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))

            # Cube Map
            cube_rgb, _ = self.e2c(rgb.copy())
            # to tensor
            images_pers = []
            images_pers_resize = []
            for i in range(cube_rgb.shape[0]):
                img_tensor = transform(cube_rgb[i])
                images_pers.append(img_tensor)
            
                img_resize_tensor = self.to_tensor(cv2.resize(cube_rgb[i], dsize=(518, 518), interpolation=cv2.INTER_NEAREST))
                images_pers_resize.append(img_resize_tensor)
            images_pers_resize = torch.stack(images_pers_resize)
            context_images = torch.stack(images_pers)

            cube_depth = torch.from_numpy(cube_depth).squeeze(-1).to(torch.float32)
            gt_depth = torch.from_numpy(gt_depth).to(torch.float32)
            # gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)
            val_mask = ((gt_depth > 0) & (gt_depth <= max_depth_meters) & ~torch.isnan(gt_depth))
            if gt_depth[val_mask].numel() == 0:
                continue

            val_mask_cube = ((cube_depth > 0) & (cube_depth <= max_depth_meters) & ~torch.isnan(cube_depth))
            # Check if any cube face is completely invalid
            if not all(torch.any(val_mask_cube[i]) for i in range(len(cube_depth))):
                continue

            # Normalize depth
            _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
            gt_depth_norm = (gt_depth - _min) / (_max - _min)
            gt_depth_norm = torch.clip(gt_depth_norm, 0.001, 1.0)

            cube_depth_norm = (cube_depth - _min) / (_max - _min)
            cube_depth_norm = torch.clip(cube_depth_norm, 0.001, 1.0)
                       

            # if 'Structured3D' not in chunk_path:
            #     continue
    
            # # Load the chunk.
            # pers_path = os.path.join(self.cfg.roots[0], chunk_path) #[0:11] + '_' + os.path.basename(self.data[idx][0]).split("_skybox")[0]

            # parts = chunk_path.split('_')
            # scene_id = parts[2]  # 00001
            # room_id = parts[3]   # 906322
            
            # # 构建路径
            # base_path = "/home/pengfei_wang/MVDiffusion/data/Structured3D"
            # depth_path = f"{base_path}/scene_{scene_id}/2D_rendering/{room_id}/panorama/full/depth.png"
            # if not os.path.exists(depth_path):
            #     print(f"Warning: {depth_path} does not exist. Skipping...")
            #     continue 
            # depth_image = Image.open(depth_path)
            # depth_array = np.array(depth_image)  # 转换为numpy数组，形状为(H, W)
            # gt_depth = torch.from_numpy(depth_array).float() / 1000
            # gt_depth[gt_depth > 10] = 10

            # val_mask = ((gt_depth > 0) & (gt_depth <= 10) & ~torch.isnan(gt_depth))
            # # valid_mask = depth_tensor != 0 
            # if gt_depth[val_mask].numel() == 0:
            #     continue

            # # Normalize depth
            # _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
            # gt_depth_norm = (gt_depth - _min) / (_max - _min)
            # gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)

            # from glob import glob
            # # png_files = sorted(glob(os.path.join(new_path, 'pers_*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
            # png_files = sorted(glob(os.path.join(pers_path, 'face_*.png')),
            #                key=lambda x: int(x.split('face_')[1].split('.png')[0]))
            
            # images_pers = []
            # for path in png_files:
            #     img = cv2.imread(path)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img_tensor = transform(img)
            #     images_pers.append(img_tensor)
            # context_images = torch.stack(images_pers)


            # W2C = extrinsics[0].clone()
            # # extrinsics_traget[..., :3, 3] = torch.tensor([0, 0, -0.1]) 
            # def convert_W2C_to_C2W(W2C):
            #     # W2C shape: [4, 4]
            #     R_W2C = W2C[:3, :3]  # shape: [3, 3]
            #     t_W2C = W2C[:3, 3]   # shape: [3]
                
            #     R_C2W = R_W2C.transpose(0, 1)
            #     t_C2W = -R_C2W @ t_W2C
                
            #     C2W = torch.eye(4, device=W2C.device)
            #     C2W[:3, :3] = R_C2W
            #     C2W[:3, 3] = t_C2W
            #     return C2W

            # def convert_C2W_to_W2C(C2W):
            #     # C2W shape: [4, 4]
            #     R_C2W = C2W[:3, :3]  # shape: [3, 3]
            #     t_C2W = C2W[:3, 3]   # shape: [3]
                
            #     R_W2C = R_C2W.transpose(0, 1)
            #     t_W2C = -R_W2C @ t_C2W
                
            #     W2C = torch.eye(4, device=C2W.device)
            #     W2C[:3, :3] = R_W2C
            #     W2C[:3, 3] = t_W2C
            #     return W2C

            # # 生成6个不同程度的zoom in
            # W2C_zoomed_list = []
            # for i in range(6):
            #     C2W = convert_W2C_to_C2W(W2C)
            #     zoom_distance = 0.1 * i
            #     C2W[:3, 3] = torch.tensor([0, 0, zoom_distance], device=W2C.device)
            #     W2C_zoomed = convert_C2W_to_W2C(C2W)
            #     W2C_zoomed_list.append(W2C_zoomed)

            # extrinsics_target = torch.stack(W2C_zoomed_list) 
            example = {
                "context": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "images_forvit": images_pers_resize,
                    "near": self.get_bound("near", context_images.shape[0]) ,
                    "far": self.get_bound("far", context_images.shape[0]),
                    "depth": gt_depth,
                    "cube_depth": cube_depth_norm,
                    "val_mask": val_mask,
                    "val_mask_cube": val_mask_cube
                },
                "target": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "images_forvit": images_pers_resize,
                    "near": self.get_bound("near", context_images.shape[0]),
                    "far": self.get_bound("far", context_images.shape[0]),
                    "depth": gt_depth,
                    "val_mask": val_mask,
                },
                "scene": chunk_path[0],
            }

            yield example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
