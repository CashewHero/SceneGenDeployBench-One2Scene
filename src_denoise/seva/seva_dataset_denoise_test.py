import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from natsort import natsorted
from omegaconf import OmegaConf
import time
from packaging import version
from PIL import Image
from argparse import ArgumentParser
import random
from seva.data_io import get_parser
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import string
from seva.modules.preprocessor import VGGTPipeline
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
import os
from pytorch_lightning.utilities import rank_zero_only
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
    get_value_dict,
    chunk_input_and_test,
    pad_indices,
    assemble,
    load_img_and_K,
    transform_img_and_K,
    get_k_from_dict
    
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
    
)



class SEVADataset_DenoiseOnline(torch.utils.data.Dataset):
    def __init__(self, VERSION_DICT, datafolder, max_cond_views, min_cond_views):
        self.VERSION_DICT = VERSION_DICT
        self.data_folder = datafolder
        self.data_paths = []
        self.max_cond_views = max_cond_views
        self.min_cond_views = min_cond_views
        
        # 更新数据路径
        self.update_data_paths()

    def update_data_paths(self):
        """更新数据路径列表，支持单个.pth文件或文件夹"""
        self.data_paths = []
        
        if self.data_folder.endswith('.pth') and os.path.isfile(self.data_folder):
            # 直接指定单个 .pth 文件
            self.data_paths.append(self.data_folder)
        elif os.path.isdir(self.data_folder):
            for scene in os.listdir(self.data_folder):
                scene_path = os.path.join(self.data_folder, scene)
                if os.path.isdir(scene_path):
                    # 查找scene目录下的data.pth文件
                    data_file = os.path.join(scene_path, 'data.pth')
                    if os.path.exists(data_file):
                        self.data_paths.append(data_file)
        else:
            raise FileNotFoundError(f'datafolder路径不存在: {self.data_folder}')

        print(f'Total video scenes found: {len(self.data_paths)}')

    def __len__(self):
        return len(self.data_paths)

    def random_cond_views(self, cond_orig_imgs, cond_Ks, cond_c2ws):
        # 随机采样m张图片
        num_view = random.randint(self.min_cond_views, self.max_cond_views)
        n = cond_orig_imgs.shape[0]
        
        # 随机选择m个索引
        selected_indices = torch.randperm(n)[:num_view]
        
        # 从原始数据中选择m张图片
        selected_imgs = cond_orig_imgs[selected_indices]
        selected_Ks = cond_Ks[selected_indices]
        selected_c2ws = cond_c2ws[selected_indices]
        
        # 有放回地采样到指定数量
        if num_view < self.num_cond_views:
            expanded_indices = torch.randint(0, num_view, (self.num_cond_views,))
            expanded_imgs = selected_imgs[expanded_indices]
            expanded_Ks = selected_Ks[expanded_indices]
            expanded_c2ws = selected_c2ws[expanded_indices]
        else:
            expanded_imgs = selected_imgs[:self.num_cond_views]
            expanded_Ks = selected_Ks[:self.num_cond_views]
            expanded_c2ws = selected_c2ws[:self.num_cond_views]
        
        return expanded_imgs, expanded_Ks, expanded_c2ws

  









    def __getitem__(self, idx):
        '''
        return dict:
        {
            'noisy_imgs': torch.Tensor, shape: [num_video_frames, H, W, 3], range: [0, 255]
            'version_dict': dict
            'scene_name': str
            'img_names': list[str], shape: [num_video_frames]
            'noisy_img_intrinsics': torch.Tensor, shape: [num_video_frames, 3, 3]
            'noisy_img_extrinsics': torch.Tensor, shape: [num_video_frames, 4, 4]
        }
        '''
        if idx % 1000 == 0:
            self.update_data_paths()

        try:
            # 加载场景数据
            scene_path = self.data_paths[idx % len(self.data_paths)]
            scene_data = torch.load(scene_path, weights_only=False, map_location='cpu')
            
            # 获取场景名称
            scene_name = os.path.basename(os.path.dirname(scene_path))
            
            # 获取所有的noisy images (rendered images)
            # 假设数据格式：cuda_images中第6张往后是noisy images
            if 'cuda_images' in scene_data:
                noisy_images = scene_data['cuda_images']  # [num_frames, 3, H, W]
                noisy_intrinsics = scene_data['intrinsics']  # [num_frames, 3, 3]
                noisy_extrinsics = scene_data['extrinsics']  # [num_frames, 4, 4]
            else:
                # 如果没有cuda_images，尝试其他可能的键名
                print(f"Warning: 'cuda_images' not found in {scene_path}")
                return self.__getitem__((idx + 1) % len(self.data_paths))
            
            # 转换图像格式：从 [num_frames, 3, H, W] 到 [num_frames, H, W, 3]
            # 并转换到 [0, 255] 范围
            noisy_imgs = noisy_images.permute(0, 2, 3, 1).contiguous().detach().cpu() * 255.0
            
            # 检查图像是否为黑色
            if noisy_imgs.max() == 0:
                print(f'Warning: {scene_path} contains black images')
                return self.__getitem__((idx + 1) % len(self.data_paths))
            
            # 创建图像名称列表
            num_frames = len(noisy_imgs)
            img_names = [f"{scene_name}_{i:06d}.png" for i in range(num_frames)]
            
            # 确保数据类型正确
            noisy_img_intrinsics = noisy_intrinsics.float()  # [num_frames, 3, 3]
            noisy_img_extrinsics = noisy_extrinsics.float()  # [num_frames, 4, 4]
            
            return {
                'noisy_imgs': noisy_imgs.clamp(0,255),  # torch.Tensor, shape: [num_video_frames, H, W, 3], range [0, 255]
                # 'version_dict': self.VERSION_DICT,  # dict
                'scene_name': scene_name,  # str
                'img_names': img_names,  # list[str], shape: [num_video_frames]
                'noisy_img_intrinsics': noisy_img_intrinsics,  # torch.Tensor, shape: [num_video_frames, 3, 3]
                'noisy_img_extrinsics': noisy_img_extrinsics,  # torch.Tensor, shape: [num_video_frames, 4, 4]
            }
                
        except Exception as e:
            print(f'Error processing scene {idx}: {e}')
            return self.__getitem__((idx + 1) % len(self.data_paths))


    def parse_task(
        self,
        task,
        scene,
        num_inputs,
        T,
        version_dict,
    ):
        options = version_dict["options"]

        anchor_indices = None
        anchor_c2ws = None
        anchor_Ks = None

        if task == "img2trajvid_s-prob":
            if num_inputs is not None:
                assert (
                    num_inputs == 1
                ), "Task `img2trajvid_s-prob` only support 1-view conditioning..."
            else:
                num_inputs = 1
            num_targets = options.get("num_targets", T - 1)
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            input_indices = [0]
            anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

            all_imgs_path = [scene] + [None] * num_targets

            c2ws, fovs = get_preset_pose_fov(
                option=options.get("traj_prior", "orbit"),
                num_frames=num_targets + 1,
                start_w2c=torch.eye(4),
                look_at=torch.Tensor([0, 0, 10]),
            )

            with Image.open(scene) as img:
                W, H = img.size
                aspect_ratio = W / H
            Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
            Ks[:, :2] *= (
                torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
            )  # normalized
            Ks = Ks.numpy()

            anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
            anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

        else:
            parser = get_parser(
                parser_type="reconfusion",
                data_dir=scene,
                normalize=False,
            )
            all_imgs_path = parser.image_paths
            c2ws = parser.camtoworlds
            camera_ids = parser.camera_ids
            Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

            if num_inputs is None:
                assert len(parser.splits_per_num_input_frames.keys()) == 1
                num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
                split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
            elif isinstance(num_inputs, str):
                split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
                num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
            else:
                split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

            num_targets = len(split_dict["test_ids"])

            if task == "img2img":
                # Note in this setting, we should refrain from using all the other camera
                # info except ones from sampled_indices, and most importantly, the order.
                num_anchors = infer_prior_stats(
                    T,
                    num_inputs,
                    num_total_frames=num_targets,
                    version_dict=version_dict,
                )

                sampled_indices = np.sort(
                    np.array(split_dict["train_ids"] + split_dict["test_ids"])
                )  # we always sort all indices first

                traj_prior = options.get("traj_prior", None)
                if traj_prior == "spiral":
                    assert parser.bounds is not None
                    anchor_c2ws = generate_spiral_path(
                        c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
                        parser.bounds[sampled_indices],
                        n_frames=num_anchors + 1,
                        n_rots=2,
                        zrate=0.5,
                        endpoint=False,
                    )[1:] @ np.diagflat([1, -1, -1, 1])
                elif traj_prior == "interpolated":
                    assert num_inputs > 1
                    anchor_c2ws = generate_interpolated_path(
                        c2ws[split_dict["train_ids"], :3],
                        round((num_anchors + 1) / (num_inputs - 1)),
                        endpoint=False,
                    )[1 : num_anchors + 1]
                elif traj_prior == "orbit":
                    c2ws_th = torch.as_tensor(c2ws)
                    lookat = get_lookat(
                        c2ws_th[sampled_indices, :3, 3],
                        c2ws_th[sampled_indices, :3, 2],
                    )
                    anchor_c2ws = torch.linalg.inv(
                        get_arc_horizontal_w2cs(
                            torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                            lookat,
                            -F.normalize(
                                c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0),
                                dim=-1,
                            ),
                            num_frames=num_anchors + 1,
                            endpoint=False,
                        )
                    ).numpy()[1:, :3]
                else:
                    anchor_c2ws = None
                # anchor_Ks is default to be the first from target_Ks

                all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
                c2ws = c2ws[sampled_indices]
                Ks = Ks[sampled_indices]

                # absolute to relative indices
                input_indices = compute_relative_inds(
                    sampled_indices,
                    np.array(split_dict["train_ids"]),
                )
                anchor_indices = np.arange(
                    sampled_indices.shape[0],
                    sampled_indices.shape[0] + num_anchors,
                ).tolist()  # the order has no meaning here

            elif task == "img2vid":
                num_targets = len(all_imgs_path) - num_inputs
                num_anchors = infer_prior_stats(
                    T,
                    num_inputs,
                    num_total_frames=num_targets,
                    version_dict=version_dict,
                )

                input_indices = split_dict["train_ids"]
                anchor_indices = infer_prior_inds(
                    c2ws,
                    num_prior_frames=num_anchors,
                    input_frame_indices=input_indices,
                    options=options,
                ).tolist()
                num_anchors = len(anchor_indices)
                anchor_c2ws = c2ws[anchor_indices, :3]
                anchor_Ks = Ks[anchor_indices]

            elif task == "img2trajvid":
                num_anchors = infer_prior_stats(
                    T,
                    num_inputs,
                    num_total_frames=num_targets,
                    version_dict=version_dict,
                )

                target_c2ws = c2ws[split_dict["test_ids"], :3]
                target_Ks = Ks[split_dict["test_ids"]]
                anchor_c2ws = target_c2ws[
                    np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
                ]
                anchor_Ks = target_Ks[
                    np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
                ]

                sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]
                all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
                c2ws = c2ws[sampled_indices]
                Ks = Ks[sampled_indices]

                input_indices = np.arange(num_inputs).tolist()
                anchor_indices = np.linspace(
                    num_inputs, num_inputs + num_targets - 1, num_anchors
                ).tolist()

            else:
                raise ValueError(f"Unknown task: {task}")

        return (
            all_imgs_path,
            num_inputs,
            num_targets,
            input_indices,
            anchor_indices,
            torch.tensor(c2ws[:, :3]).float(),
            torch.tensor(Ks).float(),
            (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
            (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
        )





if __name__ == "__main__":
    
    
    VERSION_DICT = {
        "H": 512,
        "W": 512,
        "T": 21,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest-gt",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": 2.0,
            "camera_scale": 2.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
            "num_inputs": None,
            "seed": 23
        }
    }

    mp.set_start_method('spawn', force=True)
    # vggt = VGGTPipeline(f'cuda:0')
    dataset = SEVADataset_DenoiseOnline( VERSION_DICT, datafolder='/data/liyi_chen/code/generative-models-denoise/pengfei_data/ood_demo_v2', max_cond_views=5, min_cond_views=5)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context='spawn')
    print(len(dataset))
    for i, inp_imgs in enumerate(dataloader):
        print(f'create {i} item, {inp_imgs["name"]}')
        # time.sleep(10)
        