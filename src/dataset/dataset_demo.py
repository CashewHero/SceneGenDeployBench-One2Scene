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
from .geometry import get_preset_pose_fov

def convert_W2C_to_C2W(W2C):
    if not isinstance(W2C, torch.Tensor):
        W2C = torch.from_numpy(W2C)
    # W2C shape: [..., 4, 4] 
    R_W2C = W2C[..., :3, :3]  # shape: [..., 3, 3]
    t_W2C = W2C[..., :3, 3]   # shape: [..., 3]
    
    R_C2W = R_W2C.transpose(-2, -1)  # transpose last two dimensions
    t_C2W = -R_C2W @ t_W2C.unsqueeze(-1)  # [..., 3, 1]
    t_C2W = t_C2W.squeeze(-1)  # [..., 3]
    
    C2W = torch.eye(4, device=W2C.device).expand(W2C.shape[:-2] + (4, 4)).clone()
    C2W[..., :3, :3] = R_C2W
    C2W[..., :3, 3] = t_C2W
    return C2W

def convert_C2W_to_W2C(C2W):
    if not isinstance(C2W, torch.Tensor):
        C2W = torch.from_numpy(C2W)
    # C2W shape: [..., 4, 4]
    R_C2W = C2W[..., :3, :3]  # shape: [..., 3, 3] 
    t_C2W = C2W[..., :3, 3]   # shape: [..., 3]
    
    R_W2C = R_C2W.transpose(-2, -1)  # transpose last two dimensions
    t_W2C = -R_W2C @ t_C2W.unsqueeze(-1)  # [..., 3, 1] 
    t_W2C = t_W2C.squeeze(-1)  # [..., 3]
    
    W2C = torch.eye(4, device=C2W.device).expand(C2W.shape[:-2] + (4, 4)).clone()
    W2C[..., :3, :3] = R_W2C 
    W2C[..., :3, 3] = t_W2C
    return W2C

def get_target_trajectory(preset_traj: Literal[
            "orbit",
            "spiral",
            "lemniscate",
            "zoom-in",
            "zoom-out",
            "dolly zoom-in",
            "dolly zoom-out",
            "move-forward",
            "move-backward",
            "move-up",
            "move-down",
            "move-left",
            "move-right",
        ], num_frames: int, zoom_factor: float | None,):      
    start_w2c = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]],dtype=torch.float32)
    start_c2w = torch.linalg.inv(start_w2c)
    look_at = torch.tensor([0, 0, 1])
    # look_at = torch.tensor([0, 0, 2])
    start_fov = .94
    if preset_traj == "orbit" or preset_traj == "spiral":
        spiral_radii = [0.3, 0.3, 0.1]
    else:
        spiral_radii=[1.4, 1.4, 1.0]

    # spiral_radii=[1.7, 1.7, 1.0]
    target_c2ws, target_fovs = get_preset_pose_fov(
        preset_traj,
        num_frames,
        start_w2c,
        look_at,
        -start_c2w[:3, 1],
        start_fov,
        # spiral_radii=[1.7, 1.7, 0.8],
        spiral_radii=spiral_radii,
        zoom_factor=zoom_factor,
    )
    return torch.from_numpy(target_c2ws)

def get_K_R_tensor(FOV, THETA, PHI, height, width, distance=None, scene_radius=0.1):
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



# # -------
#     # 根据场景大小设置合适的相机距离
#     if distance is None:
#     # 默认距离设置
#         optimal_distance = scene_radius * 2.0
#     else:
#         optimal_distance = distance

#     camera_position = np.array([0, 0, -optimal_distance], dtype=np.float32)
#     # 将相机位置转换到相机坐标系
#     W2C[:3, 3] = -R @ camera_position

# # -------
    
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

def move_camera_forward(W2C, distance):
    # 创建新的矩阵避免修改原始矩阵
    new_W2C = W2C.clone()
    # 设置z轴平移（相机坐标系中，z轴指向相机后方）
    new_W2C[..., :3, 3] = torch.tensor([0, 0, -distance])  # 注意是负号
    return new_W2C

# 方式2：如果你有C2W矩阵，先修改C2W再转换为W2C
def move_camera_using_C2W(C2W, distance):
    # 修改相机在世界坐标系中的位置
    new_C2W = C2W.clone()
    new_C2W[..., :3, 3] += C2W[..., :3, 2] * distance  # 沿着相机朝向移动
    # 转换回W2C
    new_W2C = torch.inverse(new_C2W)
    return new_W2C

class DatasetDemo(IterableDataset):
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

        self.w = 1024
        self.h = 512
        self.max_depth_meters = 10.0
        self.min_depth_meters = 0.01


        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.e2c = e2c
        self.root_dir = self.cfg.roots[0]

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

        
        worker_info = torch.utils.data.get_worker_info()

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


        FOV = 95
        K_tensors = []
        W2C_tensors = []

        for theta, phi in zip(u_deg, v_deg):
            K, W2C = get_K_R_tensor(FOV, theta, phi, height, width)
            K_tensors.append(K)
            W2C_tensors.append(W2C)
        intrinsics_target = torch.stack(K_tensors)

        rgb_name = os.path.join(self.root_dir, 'panorama.jpg')
        ### read panorama
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        # ERP
        erp_rgb = self.to_tensor(rgb.copy())
        # Cube Map
        cube_rgb, _ = self.e2c(rgb.copy())

        images_pers = []
        images_pers_resize = []
        for i in range(cube_rgb.shape[0]):
            img_tensor = transform(cube_rgb[i])
            images_pers.append(img_tensor)

            img_resize_tensor = self.to_tensor(cv2.resize(cube_rgb[i], dsize=(518, 518), interpolation=cv2.INTER_NEAREST))
            images_pers_resize.append(img_resize_tensor)
        images_pers_resize = torch.stack(images_pers_resize)
        context_images = torch.stack(images_pers)

        # "orbit",
        # "spiral",
        # "lemniscate",
        # "zoom-in",
        # "zoom-out",
        # "dolly zoom-in",
        # "dolly zoom-out",
        # "move-forward",
        # "move-backward",
        # "move-up",
        # "move-down",
        # "move-left",
        # "move-right",

        movements = ["orbit","move-left","lemniscate","move-forward"]
        # movements = ["move-right","move-forward1","orbit","move-right","move-forward1", "lemniscate","orbit1","move-forward"]
        # movements = ["move-forward","move-forward"]
        # movements = ["move-forwardup", "move-forwardup", "move-forwardup", "move-forwardup", "move-forwardup", "move-forwardup", "move-backward","move-backward","move-backward","move-backward","move-backward","move-backward"]


        seva_extrinsics = []
        all_movement_extrinsics_by_view = []
        number_view = 40
        start_c2w = extrinsics[0].clone()  # 当前视角的初始变换矩阵
        for movement in movements:  # 遍历所有运动
            if len(seva_extrinsics) > 0:
                start_c2w = seva_extrinsics[-1][-1].clone()

            base_target_c2ws = get_target_trajectory(movement, number_view, 0.4)  # 忽略第一个，获取目标轨迹
            target_c2ws = start_c2w @ base_target_c2ws
            seva_extrinsics.append(target_c2ws)  # 保存基础视角的结果
        seva_extrinsics = torch.cat(seva_extrinsics, dim=0)
        # all_extrinsics_target = torch.cat(base_target_c2ws, dim=0)
        all_extrinsics_target = seva_extrinsics

        # movements = ["move-backward", "move-forward", "move-up", "move-down", "move-left", "move-right"]
        # seva_extrinsics = []
        # all_movement_extrinsics_by_view = []

        # for i in range(4):  # 遍历每个视角 (0, 1, 2, 3)
        #     start_c2w = extrinsics[i].clone()  # 当前视角的初始变换矩阵
        #     movement_extrinsics = []  # 存储当前视角的所有运动结果

        #     for movement in movements:  # 遍历所有运动
        #         base_target_c2ws = get_target_trajectory(movement, 6, 1.0)[1:]  # 忽略第一个，获取目标轨迹
        #         seva_extrinsics.append(base_target_c2ws)  # 保存基础视角的结果
        #         target_c2ws = start_c2w @ base_target_c2ws  # 计算目标变换矩阵
        #         movement_extrinsics.append(target_c2ws)  # 将结果添加到当前视角的列表中

        #     # 将当前视角的所有运动拼接并保存
        #     movement_extrinsics = torch.cat(movement_extrinsics, dim=0)
        #     all_movement_extrinsics_by_view.append(movement_extrinsics)

        # # 最终按视角顺序拼接所有结果
        # all_extrinsics_target = torch.cat(all_movement_extrinsics_by_view, dim=0)
        # seva_extrinsics = torch.cat(seva_extrinsics, dim=0)

        
        # return final_extrinsics


        # W2C = extrinsics[0].clone()
        # base_target_c2ws = get_target_trajectory('move-backward', 2, 1.0)
        # all_extrinsics_target = [torch.inverse(base_target_c2ws)]  # First view's result

        # for i in range(1, len(extrinsics)):  # Start from index 1
        #     start_w2c = extrinsics[i].clone()
        #     start_c2w = torch.inverse(start_w2c)
        #     target_c2ws = base_target_c2ws @ start_c2w
        #     extrinsics_target = torch.inverse(target_c2ws)
        #     all_extrinsics_target.append(extrinsics_target)

        # # Concatenate all transformed extrinsics
        # all_extrinsics_target = torch.cat([extrinsics,final_extrinsics], dim=0)


        N = all_extrinsics_target.shape[0]
        intrinsics_target = intrinsics_target[0].clone().unsqueeze(0).repeat(N, 1, 1)

        example = {
            "context": {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "image": context_images,
                "images_forvit": images_pers_resize,
                "near": self.get_bound("near", cube_rgb.shape[0]) ,
                "far": self.get_bound("far", cube_rgb.shape[0]),
                "depth": intrinsics,
                "val_mask": intrinsics,
            },
            "target": {
                "extrinsics": all_extrinsics_target,
                "intrinsics": intrinsics_target,
                "image": context_images,
                "images_forvit": images_pers_resize,
                "near": self.get_bound("near", intrinsics_target.shape[0]),
                "far": self.get_bound("far", intrinsics_target.shape[0]),
                "depth": intrinsics,
                "val_mask": intrinsics,
                "seva_c2w": seva_extrinsics,
            },
            "scene": str(self.root_dir),
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
