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

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list

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


class Datasetmp3d(IterableDataset):
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

        if self.stage == "train":
            self.chunks = read_list("/home/pengfei_wang/NoPoSplat/src/dataset_read/datasets/matterport3d_train.txt")
        elif self.stage == "val":
            self.chunks = read_list("/home/pengfei_wang/NoPoSplat/src/dataset_read/datasets/matterport3d_test.txt")
        elif self.stage == "test":
            self.chunks = read_list("/home/pengfei_wang/NoPoSplat/src/dataset_read/datasets/matterport3d_test.txt")
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        


        self.w = 1024
        self.h = 512
        self.max_depth_meters = 10.0
        self.min_depth_meters = 0.01


        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.e2c = e2c
        self.root_dir = self.cfg.roots[0]
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

            rgb_name = os.path.join(self.root_dir, chunk_path[0])
            # rgb_name = rgb_name.replace("rgb_", "rgb_pre_") # use the preprocessed rgb images
            rgb = cv2.imread(rgb_name)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

            depth_name = os.path.join(self.root_dir, chunk_path[1])
            gt_depth = cv2.imread(depth_name, -1)
            gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
            gt_depth = gt_depth.astype(float)/4000
            gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1
            # gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters

            # gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)
            gt_depth = torch.from_numpy(gt_depth).to(torch.float32)[None]

            # rgb = self.normalize(rgb)

            # # Disparity
            # gt_disp = gt_depth.copy()
            # gt_disp[gt_disp > 0] = 1.0 / gt_disp[gt_disp > 0]
            # gt_disp = gt_disp.astype(np.float32)
            # gt_disp = torch.from_numpy(np.expand_dims(gt_disp, axis=0)).to(torch.float32)


            # ERP
            erp_rgb = self.to_tensor(rgb.copy())
            # Cube Map
            cube_rgb, _ = self.e2c(rgb.copy())

            # to tensor
            # cube_rgb = self.to_tensor(cube_rgb)
            images_pers = []
            images_pers_resize = []
            for i in range(cube_rgb.shape[0]):
                img_tensor = transform(cube_rgb[i])
                images_pers.append(img_tensor)

                img_resize_tensor = self.to_tensor(cv2.resize(cube_rgb[i], dsize=(518, 518), interpolation=cv2.INTER_NEAREST))
                images_pers_resize.append(img_resize_tensor)
            images_pers_resize = torch.stack(images_pers_resize)
            context_images = torch.stack(images_pers)

            rgb = self.to_tensor(rgb.copy())
            # cube_gt = self.e2c.run(np.expand_dims(gt_disp.copy(), axis=2))

            val_mask = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters)
                                & ~torch.isnan(gt_depth))

            #    # Normalize depth
            # _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
            # gt_depth_norm = (gt_depth - _min) / (_max - _min)
            # gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)
            rgb_val_mask = (rgb > 0)[0].unsqueeze(0)

            example = {
                "context": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "images_forvit": images_pers_resize,
                    "near": self.get_bound("near", cube_rgb.shape[0]),
                    "far": self.get_bound("far", cube_rgb.shape[0]),
                    "depth": gt_depth,
                    "val_mask": val_mask,
                    "rgb_val_mask": rgb_val_mask,
                },
                "target": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "images_forvit": images_pers_resize,
                    "near": self.get_bound("near", cube_rgb.shape[0]),
                    "far": self.get_bound("far", cube_rgb.shape[0]),
                    "depth": gt_depth,
                    "val_mask": val_mask,
                    "rgb_val_mask": rgb_val_mask,
                },
                "scene": chunk_path,
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
