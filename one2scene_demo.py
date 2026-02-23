#!/usr/bin/env python3
"""
从 train.py 改造的推理脚本，直接使用 trainer.test()
"""

import os
import sys
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


class PanoramaDataModule:
    """自定义数据模块，用于加载单张全景图"""
    
    def __init__(self, pano_image_path: str, cube_size: int = 512):
        self.pano_image_path = pano_image_path
        self.cube_size = cube_size
        self.batch = None
    
    def setup(self, stage=None):
        """准备数据"""
        if stage == "test" or stage is None:
            self.batch = self._prepare_batch()
    
    def _prepare_batch(self):
        """准备全景图 batch"""
        import cv2
        import numpy as np
        from torchvision import transforms
        
        # 导入转换工具
        from test_step_demo import EquirectangularToCubemap, get_K_R_tensor
        
        print(f"📷 Loading panorama: {self.pano_image_path}")
        
        # 加载全景图
        pano_img = cv2.imread(self.pano_image_path)
        if pano_img is None:
            raise ValueError(f"Cannot load image from {self.pano_image_path}")
        
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
        
        # 转换为 cubemap
        e2c_converter = EquirectangularToCubemap(cube_size=self.cube_size)
        cube_faces = e2c_converter.convert(pano_img)
        
        # 转换为 tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        cube_tensors = []
        for face in cube_faces:
            face_tensor = transform(face)
            cube_tensors.append(face_tensor)
        
        context_images = torch.stack(cube_tensors).unsqueeze(0)  # [1, 6, 3, H, W]
        
        # 生成相机参数
        u_deg = [0, 90, 180, 270, -90, -90]
        v_deg = [0, 0, 0, 0, 90, -90]
        FOV = 95
        
        K_list = []
        extrinsics_list = []
        
        for theta, phi in zip(u_deg, v_deg):
            K, W2C = get_K_R_tensor(FOV, theta, phi, self.cube_size, self.cube_size)
            K_list.append(K)
            extrinsics_list.append(W2C)
        
        intrinsics = torch.stack(K_list).unsqueeze(0)  # [1, 6, 3, 3]
        extrinsics = torch.stack(extrinsics_list).unsqueeze(0)  # [1, 6, 4, 4]
        
        # 生成 target 视角
        target_views = self._generate_target_views(self.cube_size, num_views=140)
        
        batch = {
            "context": {
                "image": context_images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "near": torch.ones(1, 6, 1) * 0.1,
                "far": torch.ones(1, 6, 1) * 100.0,
            },
            "target": {
                "image": target_views["images"],
                "intrinsics": target_views["intrinsics"],
                "extrinsics": target_views["extrinsics"],
                "seva_c2w": target_views["extrinsics"],
                "near": torch.ones(1, target_views["num_views"], 1) * 0.1,
                "far": torch.ones(1, target_views["num_views"], 1) * 100.0,
            },
            "scene": [Path(self.pano_image_path).stem],
        }
        
        return batch
    
    def _generate_target_views(self, cube_size: int, num_views: int = 100):
        """生成目标渲染视角"""
        from test_step_demo import get_K_R_tensor
        
        theta_list = np.linspace(0, 360, num_views, endpoint=False)
        phi_list = np.zeros(num_views)
        
        K_list = []
        extrinsics_list = []
        FOV = 90
        
        for theta, phi in zip(theta_list, phi_list):
            K, W2C = get_K_R_tensor(FOV, theta, phi, cube_size, cube_size)
            K_list.append(K)
            extrinsics_list.append(W2C)
        
        intrinsics = torch.stack(K_list).unsqueeze(0)
        extrinsics = torch.stack(extrinsics_list).unsqueeze(0)
        images = torch.zeros(1, num_views, 3, cube_size, cube_size)
        
        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "num_views": num_views
        }
    
    def test_dataloader(self):
        """返回测试数据加载器"""
        return [self.batch]


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def inference(cfg_dict: DictConfig):
    """推理主函数"""
    
    # 加载配置
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # 设置输出目录
    output_dir = Path(cfg_dict.test.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Output directory: {output_dir}")
    
    # 设置 logger
    logger = LocalLogger()
    
    # Checkpoint 路径
    checkpoint_path = cfg_dict.checkpointing.load
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    # Step tracker（测试时可以为 None）
    step_tracker = None
    
    # 创建 Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        enable_progress_bar=True,
        inference_mode=True,
    )
    
    # 设置随机种子
    torch.manual_seed(cfg_dict.seed)
    
    # 获取 encoder 和 decoder
    print("🏗️ Building model...")
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    
    # 创建 ModelWrapper
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        decoder,
        get_losses(cfg.loss),
        step_tracker,
        distiller=None,
    )
    
    print("✓ Model created")
    
    # 创建数据模块
    print("📷 Preparing data...")
    pano_image_path = cfg_dict.test.get("pano_image_path", None)
    
    if pano_image_path is None:
        # 如果没有指定全景图，使用原始数据模块
        print("  Using original DataModule")
        data_module = DataModule(
            cfg.dataset,
            cfg.data_loader,
            step_tracker,
            global_rank=trainer.global_rank,
        )
    else:
        # 使用自定义全景图数据模块
        print(f"  Using panorama: {pano_image_path}")
        data_module = PanoramaDataModule(
            pano_image_path=pano_image_path,
            cube_size=cfg_dict.test.get("cube_size", 512)
        )
    
    print("✓ Data prepared")
    
    # 运行测试
    print("\n🎯 Running inference...")
    print("=" * 80)
    
    trainer.test(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )
    
    print("=" * 80)
    print(f"✅ Done! Results saved to {output_dir}")


if __name__ == "__main__":
    inference()