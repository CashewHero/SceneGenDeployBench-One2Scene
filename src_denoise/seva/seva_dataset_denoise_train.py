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
    def __init__(self, rank_id, VERSION_DICT, datafolder,max_cond_views, min_cond_views, 
                 num_cond_noisy_view, num_cond_clear_view, num_target_view):
        self.rank_id = rank_id
        self.VERSION_DICT = VERSION_DICT
        self.num_views = 21 # VERSION_DICT['T']
        self.num_cond_noisy_view = num_cond_noisy_view
        self.num_cond_clear_view = num_cond_clear_view
        self.num_target_view = num_target_view
        self.data_folder = datafolder
        self.data_paths = []
        self.update_data_paths()
        self.max_cond_views = max_cond_views
        self.min_cond_views = min_cond_views

    def update_data_paths(self):
        self.data_paths = []
        for scene in os.listdir(self.data_folder):
            self.data_paths.append(os.path.join(self.data_folder, scene))
        print(f'total data paths: {len(self.data_paths)}')

    def __len__(self):
        return 1000000000

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
        
        # 有放回地采样n次
        expanded_indices = torch.randint(0, num_view, (n,)) 

        # 扩充到原始大小
        expanded_imgs = selected_imgs[expanded_indices]
        expanded_Ks = selected_Ks[expanded_indices]
        expanded_c2ws = selected_c2ws[expanded_indices]
        
        return expanded_imgs, expanded_Ks, expanded_c2ws

    def __getitem__(self, idx):
        # ====set fix seed======
        # idx = 105
        # torch.manual_seed(105)
        # random.seed(105)
        # np.random.seed(105)
        # ==========================
        gradio=False

        num_cond_noisy_view = self.num_cond_noisy_view
        num_cond_clear_view = self.num_cond_clear_view
        num_target_view = self.num_target_view
        assert num_cond_clear_view  + num_target_view == self.num_views or num_cond_noisy_view + num_cond_clear_view  + num_target_view == self.num_views
        if idx % 1000 == 0:
            self.update_data_paths()
            
        task = 'img2img'
        try:
            scene_data = torch.load(self.data_paths[idx % len(self.data_paths)], weights_only=True, map_location='cpu')
        except Exception as e:
            print(f'{self.data_paths[idx % len(self.data_paths)]} is corrupted')
            return self.__getitem__(idx+1)
        cond_orig_imgs = scene_data['context']['image'][0].permute(0,2,3,1).contiguous().detach().cpu()*255.0 # [f_cond,  H, W, 3] with rage(0, 255)
        noisy_target_imgs = scene_data['target']['rendered_image'][0].detach().permute(0,2,3,1).contiguous().detach().cpu()*255.0 # [f_target,  H, W, 3] with rage(0, 255)
        target_imgs = scene_data['target']['image'][0].permute(0,2,3,1).contiguous().detach().cpu()*255.0 # [f_target, H, W, 3]
        
        source_view_mask = torch.cat([torch.ones(num_cond_noisy_view), torch.zeros(num_cond_clear_view + num_target_view)], dim=0)
        source_view_mask = source_view_mask[:, None, None, None]
        
        # 为了避免DDP检测到LoRA参数未使用，确保source_view_mask中源视图和目标视图都有合理分布
        # 随机调整一些目标视图为源视图，确保两套LoRA都会被使用
        if self.training if hasattr(self, 'training') else True:  # 只在训练时进行随机化
            total_views = len(source_view_mask)
            # 确保至少有25%的视图是源视图，25%是目标视图
            min_source_views = max(1, total_views // 4)
            min_target_views = max(1, total_views // 4)
            
            # 当前源视图数量
            current_source_count = source_view_mask.sum().item()
            
            if current_source_count < min_source_views:
                # 需要增加源视图，随机将一些目标视图改为源视图
                target_indices = torch.where(source_view_mask.squeeze() == 0)[0]
                if len(target_indices) > 0:
                    num_to_flip = min(min_source_views - current_source_count, len(target_indices))
                    flip_indices = target_indices[torch.randperm(len(target_indices))[:num_to_flip]]
                    source_view_mask[flip_indices] = 1
            elif total_views - current_source_count < min_target_views:
                # 需要增加目标视图，随机将一些源视图改为目标视图  
                source_indices = torch.where(source_view_mask.squeeze() == 1)[0]
                if len(source_indices) > 0:
                    num_to_flip = min(min_target_views - (total_views - current_source_count), len(source_indices))
                    flip_indices = source_indices[torch.randperm(len(source_indices))[:num_to_flip]]
                    source_view_mask[flip_indices] = 0
        
        if noisy_target_imgs.max() == 0:
            print(f'{self.data_paths[idx % len(self.data_paths)]} rendred view is black image')
            return self.__getitem__(idx+1)
        cond_Ks = scene_data['context']['intrinsics'][0].cpu()  # [f_cond, 3, 3]
        target_Ks = scene_data['target']['intrinsics'][0].cpu() # [f_target, 3, 3]
        cond_c2ws = scene_data['context']['extrinsics'][0].cpu() # [f_cond, 4, 4]
        target_c2ws = scene_data['target']['extrinsics'][0].cpu() # [f_target, 4, 4]
        
        # reduce the number of conditional views to improve the effect of noisy_target_imgs
        if self.max_cond_views > 0 and self.min_cond_views > 0:
            cond_orig_imgs, cond_Ks, cond_c2ws = self.random_cond_views(cond_orig_imgs, cond_Ks, cond_c2ws)
        else:
            assert False, 'max_cond_views and min_cond_views should be set'
        
        # ===================================================
        # 1. random select ${num_target_view} target view and noisy_target_imgs 
        target_indices = torch.randperm(len(target_imgs))[:num_target_view]
        target_imgs = target_imgs[target_indices]
        noisy_target_imgs = noisy_target_imgs[target_indices]
        target_Ks = target_Ks[target_indices]
        target_c2ws = target_c2ws[target_indices]
        # 2. noisy_target_imgs as part of cond view
        cond_orig_imgs = torch.cat([noisy_target_imgs,cond_orig_imgs], dim=0)
        cond_Ks = torch.cat([target_Ks, cond_Ks], dim=0)
        cond_c2ws = torch.cat([target_c2ws, cond_c2ws], dim=0)
        # =======================================================================   

        orig_imgs = torch.cat([cond_orig_imgs, target_imgs], dim=0).numpy() # target_imgs 是 (8noisy_img+5condview+8target_view)*576*576*3
        noisy_target_imgs = torch.cat([cond_orig_imgs*0, noisy_target_imgs], dim=0).numpy() # noisy_target_imgs 也是 (8noisy_img+5condview+8target_view)*576*576*3
        vggt_Ks = torch.cat([cond_Ks, target_Ks], dim=0).contiguous()
        vggt_Ks[:, 0:1] = vggt_Ks[:, 0:1] * target_imgs.shape[2] # recovery to unnormalized K
        vggt_Ks[:, 1:2] = vggt_Ks[:, 1:2] * target_imgs.shape[1]
        vggt_c2ws = torch.cat([cond_c2ws, target_c2ws], dim=0).contiguous()


        options = self.VERSION_DICT["options"]
        H, W, T, C, F, options = (
            self.VERSION_DICT["H"],
            self.VERSION_DICT["W"],
            self.VERSION_DICT["T"],
            self.VERSION_DICT["C"],
            self.VERSION_DICT["f"],
            self.VERSION_DICT["options"],
        )
        # 获取数据


        num_inputs = len(cond_orig_imgs)
        num_targets = len(orig_imgs) - num_inputs
        input_indices = [ℹ for i in range(num_inputs)]
        anchor_indices = [i for i in range(len(orig_imgs)) if i not in input_indices] # anchor就是target
        c2ws = vggt_c2ws.float()
        Ks = vggt_Ks.float()
        anchor_c2ws = torch.stack([vggt_c2ws[i] for i in anchor_indices]).float()
        anchor_Ks = torch.stack([vggt_Ks[i] for i in anchor_indices]).float()

 
        traj_prior_c2ws = anchor_c2ws
        traj_prior_Ks = anchor_Ks

        # Create image conditioning.
        image_cond = {
            "img": orig_imgs,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        # Create camera conditioning.
        camera_cond = {
            "c2w": c2ws.clone(),
            "K": Ks.clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }
        
        if isinstance(image_cond, str):
            image_cond = {"img": [image_cond]}
        imgs_clip, imgs, img_size = [], [], None

        for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])): # 对cond和target图像进行resize和crop，还有camera pose也做相应修改
            if isinstance(img, str) or img is None:
                img, K = load_img_and_K(img or img_size, None, K=K, device="cpu")  # type: ignore
                img_size = img.shape[-2:]
                if options.get("L_short", -1) == -1:
                    img, K = transform_img_and_K(
                        img,
                        (W, H),
                        K=K[None],
                        mode=(
                            options.get("transform_input", "crop")
                            if i in image_cond["input_indices"]
                            else options.get("transform_target", "crop")
                        ),
                        scale=(
                            1.0
                            if i in image_cond["input_indices"]
                            else options.get("transform_scale", 1.0)
                        ),
                    )
                else:
                    downsample = 3
                    assert options["L_short"] % F * 2**downsample == 0, (
                        "Short side of the image should be divisible by "
                        f"F*2**{downsample}={F * 2**downsample}."
                    )
                    img, K = transform_img_and_K(
                        img,
                        options["L_short"],
                        K=K[None],
                        size_stride=F * 2**downsample,
                        mode=(
                            options.get("transform_input", "crop")
                            if i in image_cond["input_indices"]
                            else options.get("transform_target", "crop")
                        ),
                        scale=(
                            1.0
                            if i in image_cond["input_indices"]
                            else options.get("transform_scale", 1.0)
                        ),
                    )
                    version_dict["W"] = W = img.shape[-1]
                    version_dict["H"] = H = img.shape[-2]
                K = K[0]
                K[0] /= W
                K[1] /= H
                camera_cond["K"][i] = K
                img_clip = img
            elif isinstance(img, np.ndarray):
                img_size = torch.Size(img.shape[:2])
                img = torch.as_tensor(img).permute(2, 0, 1).contiguous()
                img = img.unsqueeze(0)
                img = img / 255.0 * 2.0 - 1.0
                if not gradio:
                    img, K = transform_img_and_K(img, (W, H), K=K[None])
                    assert K is not None
                    K = K[0]
                K[0] /= W
                K[1] /= H
                camera_cond["K"][i] = K
                img_clip = img
            else:
                assert (
                    False
                ), f"Variable `img` got {type(img)} type which is not supported!!!"
            imgs_clip.append(img_clip)
            imgs.append(img)
        imgs_clip = torch.cat(imgs_clip, dim=0)
        imgs = torch.cat(imgs, dim=0)

        
        noisy_imgs = []
        for i, (noisy_img, K) in enumerate(zip(noisy_target_imgs, camera_cond["K"])):
            if isinstance(noisy_img, np.ndarray):
                img_size = torch.Size(noisy_img.shape[:2])
                img = torch.as_tensor(noisy_img).permute(2, 0, 1).contiguous()
                img = img.unsqueeze(0)
                img = img / 255.0 * 2.0 - 1.0
                if not gradio:
                    img, _ = transform_img_and_K(img, (W, H), K=K[None])
                noisy_imgs.append(img)
        noisy_imgs = torch.cat(noisy_imgs, dim=0)

        # liyi todo filter hard example
        # 1. 计算noisy_imgs和target_imgs的ssim
        # error = torch.nn.functional.mse_loss(noisy_imgs, target_imgs, reduction='none').mean()
        # print(f'error: {error}')
        # if error>0.4:
        #     return self.__getitem__(idx+1)

        if traj_prior_Ks is not None:
            assert img_size is not None
            for i, prior_k in enumerate(traj_prior_Ks):
                img, prior_k = load_img_and_K(img_size, None, K=prior_k, device="cpu")  # type: ignore
                img, prior_k = transform_img_and_K(
                    img,
                    (W, H),
                    K=prior_k[None],
                    mode=options.get(
                        "transform_target", "crop"
                    ),  # mode for prior is always same as target
                    scale=options.get(
                        "transform_scale", 1.0
                    ),  # scale for prior is always same as target
                )
                prior_k = prior_k[0]
                prior_k[0] /= W
                prior_k[1] /= H
                traj_prior_Ks[i] = prior_k


        # Get Data
        input_indices = image_cond["input_indices"]
        input_imgs = imgs[input_indices]
        input_imgs_clip = imgs_clip[input_indices]
        input_c2ws = camera_cond["c2w"][input_indices]
        input_Ks = camera_cond["K"][input_indices]

        test_indices = [i for i in range(len(imgs)) if i not in input_indices]
        test_imgs = imgs[test_indices]
        test_imgs_clip = imgs_clip[test_indices]
        test_c2ws = camera_cond["c2w"][test_indices]
        test_Ks = camera_cond["K"][test_indices]

        assert traj_prior_c2ws is not None, (
            "`traj_prior_c2ws` should be set when using 2-pass sampling. One "
            "potential reason is that the amount of input frames is larger than "
            "T. Set `num_prior_frames` manually to overwrite the infered stats."
        )
        traj_prior_c2ws = torch.as_tensor(
            traj_prior_c2ws,
            device=input_c2ws.device,
            dtype=input_c2ws.dtype,
        )

        if traj_prior_Ks is None:
            traj_prior_Ks = test_Ks[:1].repeat_interleave(
                traj_prior_c2ws.shape[0], dim=0
            )

        # 为了训练，我们不能把traj_prior_imgs设置为0，而是由train_step时添加随机噪声
        # traj_prior_imgs = imgs.new_zeros(traj_prior_c2ws.shape[0], *imgs.shape[1:])
        # traj_prior_imgs_clip = imgs_clip.new_zeros(
        #     traj_prior_c2ws.shape[0], *imgs_clip.shape[1:]
        # )
        traj_prior_imgs = test_imgs
        traj_prior_imgs_clip = test_imgs_clip

        # ---------------------------------- first pass ----------------------------------
        T_first_pass = self.num_views # T[0] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get(
            "chunk_strategy_first_pass", "gt-nearest"
        )
        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            prior_inds_per_chunk,
            prior_sels_per_chunk,
        ) = chunk_input_and_test(
            T_first_pass,
            input_c2ws,
            traj_prior_c2ws,
            input_indices,
            image_cond["prior_indices"],
            options=options,
            task=task,
            chunk_strategy=chunk_strategy_first_pass,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )

        # print(
        #     f"Two passes (first) - chunking with `{chunk_strategy_first_pass}` strategy: total "
        #     f"{len(input_inds_per_chunk)} forward(s) ..."
        # )
        
        all_samples = {}
        all_prior_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_prior_inds,
            chunk_prior_sels,
        ) in enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    prior_inds_per_chunk,
                    prior_sels_per_chunk,
                )
            ):
            (
                curr_input_sels,
                curr_prior_sels,
                curr_input_maps,
                curr_prior_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_prior_sels,
                T=T_first_pass,
                padding_mode=options.get("t_padding_mode", "last"),
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_prior_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_prior_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                input_imgs,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                input_imgs_clip,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat([input_c2ws, traj_prior_c2ws[all_prior_inds]], dim=0),
                        torch.cat([input_Ks, traj_prior_Ks[all_prior_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [
                        traj_prior_imgs,
                        traj_prior_imgs_clip,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                )
            ]
            
            value_dict = get_value_dict(
                curr_imgs,
                curr_imgs_clip,
                curr_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_first_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=options.get("camera_scale", 2.0),
            )
            # editing paired data
            # edited_imgs, edited_prompt = self.gemini.edit_mvimgs(curr_imgs)
            # if edited_imgs is None or edited_prompt is None:
            #     return self.__getitem__(idx)
            # value_dict['edited_imgs'] = edited_imgs
            # masked paired data
            value_dict['edited_imgs'] = noisy_imgs
            value_dict['source_view_mask'] = source_view_mask


            torch.cuda.empty_cache()
            return value_dict


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
    dataset = SEVADataset_DenoiseOnline(0, VERSION_DICT, '/dfs/dataset/mvsplat360_noisyimg960p', max_cond_views=1, min_cond_views=1, num_cond_noisy_view=8, num_cond_clear_view=5, num_target_view=8)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1,  multiprocessing_context='spawn')
    print(len(dataset))
    for i, inp_imgs in enumerate(dataloader):
        print(f'create {i} item')
        # time.sleep(10)
        


