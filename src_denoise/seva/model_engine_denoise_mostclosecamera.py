import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import pytorch_lightning as pl
import torch
import numpy as np
from torchvision.utils import save_image
from omegaconf import ListConfig, OmegaConf
import os
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
import random
import deepspeed
import wandb
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.modules.ema import LitEma
from sgm.util import (default, disabled_train, get_obj_from_str, append_dims,
                    instantiate_from_config, log_txt_as_img)
from pytorch_lightning.utilities import rank_zero_only
from seva.ssim_psnr import calculate_ssim_pt, calculate_psnr_pt
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
    get_camera_dist,
)
from seva.model_denoise import SGMWrapper as SGMWrapper
from seva.utils_denoise import load_model
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.model_conditioner import SevaConditioner
from seva.eval import create_samplers



class SevaEngine(pl.LightningModule):
    def __init__(
        self,
        VERSION_DICT,
        network_config,
        baseline_denoiser_config,
        denoiser_config,
        first_stage_config,
        sigma_sampler_config,
        loss_weighting_config,
        offset_noise_level=0.0,
        loss_type="l2",
        ucg_rate=0.0,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "cond_frames",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        test_data_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        save_dir: str = "/home/pengfei_wang/One2Scene/demo_outputs/render_denoise",
    ):
        super().__init__()

        self.ucg_rate = ucg_rate
        self.save_dir = save_dir
        self.test_data_config = test_data_config
        self.MODEL = SGMWrapper(load_model(device="cpu", verbose=True)).to(torch.float32)

        # for name, param in self.MODEL.named_parameters():
        #     if not name.__contains__('condition_sourceview_module'):
        #         param.requires_grad = False
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.


        self.BASELINE_DENOISER = instantiate_from_config(baseline_denoiser_config)
        self.DENOISER = instantiate_from_config(denoiser_config)
        self.VERSION_DICT = VERSION_DICT
        self.options = self.VERSION_DICT["options"]
        self._init_seva_conditioner()
        # self._init_vggt()
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.offset_noise_level = offset_noise_level    
        self.loss_type = loss_type
        self.sampler = create_samplers(
                self.options["guider_types"],
                self.DENOISER.discretization,
                [self.VERSION_DICT['T']],
                self.options["num_steps"],
                self.options["cfg_min"],
                abort_event=None,
            )[0]
        #############

        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )

        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        # self.loss_fn = (
        #     instantiate_from_config(loss_fn_config)
        #     if loss_fn_config is not None
        #     else None
        # )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        # 添加统计字典
        self.statistics = {
            'sigma_losses': {},  # 存储每个sigma值的loss列表
            'sigma_means': {},   # 存储每个sigma值的平均loss
            'total_samples': 0   # 总样本数
        }
        
        # 初始化sigma值的统计
        for i in range(1, 84, 5):
            self.statistics['sigma_losses'][i] = []
            self.statistics['sigma_means'][i] = 0.0

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        model = instantiate_from_config(config)
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def _init_seva_conditioner(self):
        AE = AutoEncoder(chunk_size=1).eval()
        CONDITIONER = CLIPConditioner().eval()
        for param in AE.parameters():
            param.requires_grad = False
        for param in CONDITIONER.parameters():
            param.requires_grad = False
        self.SEVA_CONDITIONER = SevaConditioner(AE, CONDITIONER, self.VERSION_DICT, self.options["encoding_t"])


    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key].contiguous()

    # @torch.no_grad()
    def decode_first_stage(self, z):
        # z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        for n in range(n_rounds):
            if isinstance(self.first_stage_model.decoder, VideoDecoder):
                kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            else:
                kwargs = {}
            # out = self.first_stage_model.decode(
            #     z[n * n_samples : (n + 1) * n_samples], **kwargs
            # )
            out = deepspeed.checkpointing.checkpoint(self.SEVA_CONDITIONER.ae.decode,
                z[n * n_samples : (n + 1) * n_samples], self.options["decoding_t"]
            )
            all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        for n in range(n_rounds):
            out = self.SEVA_CONDITIONER.ae.encode(
                    x[n * n_samples : (n + 1) * n_samples], self.options["encoding_t"]
                    )
            all_out.append(out)
        z = torch.cat(all_out, dim=0)
        # z = self.scale_factor * z
        return z



    def visualize_denoised_results(self, noise, sigmas, x_target_latent, c, additional_model_inputs):
        with torch.inference_mode():
            for i in range(1, 84, 5):
                sigmas_bc = append_dims(torch.ones_like(sigmas).to(x_target_latent)* i, x_target_latent.ndim)
                noised_input = self.get_noised_input(sigmas_bc, noise, x_target_latent)
                model_output = self.DENOISER(
                    self.MODEL, noised_input, sigmas, c, **additional_model_inputs
                )
                
                # save the model_output
                loss = self.get_loss(model_output, x_target_latent, 1)
                print(f'when sigma: {i}, loss: {loss.mean()}')
                # rgb = self.decode_first_stage(model_output)
                # rgb.save(f'model_output_sigma{i}.png')

    def noisy_imgs_Ks_c2ws_cond_imgs_K2_c2ws_2_batch(self, noisy_views, noisy_Ks, noisy_c2ws, cond_views, cond_Ks, cond_c2ws, version_dict):
        """
        package the images, intrinsics and extrinsics into a batch according to version_dict
        args:
            noisy_views: the raw noisy images 
                torch.Tensor. shape: [num_noisy_views, 3, h, w], range: [0, 255]
            noisy_K2: unnormalized intrinsics
                torch.Tensor. shape: [num_noisy_views, 3, 3]
            noisy_c2ws: unnormalized extrinsics
                torch.Tensor. shape: [num_noisy_views, 4, 4]
            cond_views: the raw condition images 
                torch.Tensor. shape: [num_noisy_views, 3, h, w], range: [0, 255]
            cond_K2: unnormalized intrinsics
                torch.Tensor. shape: [num_noisy_views, 3, 3]
            cond_c2ws: unnormalized extrinsics
                torch.Tensor. shape: [num_noisy_views, 4, 4]
            version_dict: the information about height, width
        return: 
            dict: the informantion used to train
                {
                    'cond_frames_without_noise': torch.Tensor [num_noisy_view+num_cond_view, 3, 512, 512], range [-1, 1]
                    'cond_frames': torch.Tensor [num_noisy_view+num_cond_view+num_target_view, 3, 512, 512], range [-1, 1]
                    'cond_frames_mask': torch.Tensor, torch.bool, shape: [num_noisy_view+num_cond_view+num_target_view]
                    'cond_aug': torch.float64, shape: [1] 
                    'plucker_coordinate': torch.Tensor, shape: [num_noisy_view+num_cond_view+num_target_view, 6 ,h//8, w//8]
                    'c2w': torch.Tensor, shape: [num_noisy_view+num_cond_view+num_target_view, 4, 4]
                    'K': torch.Tensor, shape: [num_noisy_view+num_cond_view+num_target_view, 3, 3]
                    'camera_mask': torch.Tensor, torch.bool, all True, shape: [num_noisy_view+num_cond_view+num_target_view]
                    'source_view_mask': torch.Tensor, torch.float32, shape: [num_noisy_view+num_cond_view+num_target_view, 1, 1, 1], value: {0, 1}
                    'edited_imgs': to be delete
                    'scene_name': str
                    'image_name': list[str] each image name
                }
        """
        import numpy as np
        
        # 保存原始设备信息和数据类型
        device = noisy_views.device
        
        # 转换所有torch tensor输入为numpy（按照原来的设计）
        noisy_views_np = noisy_views.detach().cpu().numpy()
        noisy_Ks_np = noisy_Ks.detach().cpu().numpy()
        noisy_c2ws_np = noisy_c2ws.detach().cpu().numpy()
        cond_views_np = cond_views.detach().cpu().numpy()
        cond_Ks_np = cond_Ks.detach().cpu().numpy()
        cond_c2ws_np = cond_c2ws.detach().cpu().numpy()
        
        # 使用原始numpy版本的逻辑
        gradio=False
        task = 'img2img'
        H, W, T, C, F, options = (
            version_dict["H"],
            version_dict["W"],
            version_dict["T"],
            version_dict["C"],
            version_dict["f"],
            version_dict["options"],
        )
        num_cond = len(cond_views_np)
        num_noisy = len(noisy_views_np)
        num_inputs = num_noisy + num_cond  
        num_targets = num_noisy
        assert num_inputs + num_targets == T, f'the total number of views must match the expected number: {num_inputs}+{num_targets}!={T}'

        # 构造最终顺序：8个noisy + cond views + 复制的8个noisy
        final_imgs = np.concatenate([noisy_views_np, cond_views_np, noisy_views_np], axis=0)
        final_Ks = np.concatenate([noisy_Ks_np, cond_Ks_np, noisy_Ks_np], axis=0)
        final_c2ws = np.concatenate([noisy_c2ws_np, cond_c2ws_np, noisy_c2ws_np], axis=0)
        
        # 恢复K到非归一化状态
        final_Ks[:, 0:1] = final_Ks[:, 0:1] * cond_views_np.shape[-1]
        final_Ks[:, 1:2] = final_Ks[:, 1:2] * cond_views_np.shape[-2]

        # 设置索引
        input_indices = list(range(num_inputs))  # 前8个noisy + cond views作为input
        anchor_indices = list(range(num_inputs, len(final_imgs)))  # 后8个noisy作为target
        
        # 转换回torch tensor（先在CPU上处理）
        c2ws = torch.from_numpy(final_c2ws).float()
        Ks = torch.from_numpy(final_Ks).float()
        anchor_c2ws = torch.from_numpy(np.stack([final_c2ws[i] for i in anchor_indices])).float()
        anchor_Ks = torch.from_numpy(np.stack([final_Ks[i] for i in anchor_indices])).float()
        
        traj_prior_c2ws = anchor_c2ws
        traj_prior_Ks = anchor_Ks

        # Create image conditioning.
        image_cond = {
            "img": final_imgs,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        
        # Create camera conditioning.
        camera_cond = {
            "c2w": c2ws.clone(),
            "K": Ks.clone(),
            "input_indices": list(range(len(final_imgs))),
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
                        K=K[None] if K is not None else None,
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
                        K=K[None] if K is not None else None,
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
                    self.VERSION_DICT["W"] = W = img.shape[-1]
                    self.VERSION_DICT["H"] = H = img.shape[-2]
                if K is not None:
                    K = K[0]
                    K[0] /= W
                    K[1] /= H
                    camera_cond["K"][i] = K
                img_clip = img
            elif isinstance(img, np.ndarray):
                img_size = torch.Size(img.shape[-2:])
                img = torch.as_tensor(img)# .permute(2, 0, 1).contiguous()
                img = img.unsqueeze(0)
                img = img / 255.0 * 2.0 - 1.0
                if not gradio:
                    img, K = transform_img_and_K(img, (W, H), K=K[None] if K is not None else None)
                    if K is not None:
                        K = K[0]
                if K is not None:
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
        # Convert everything to device at the end
        for i in range(len(camera_cond["K"])):
            if camera_cond["K"][i] is not None:
                camera_cond["K"][i] = camera_cond["K"][i].to(device)
        
        imgs_clip = torch.cat(imgs_clip, dim=0).to(device)
        imgs = torch.cat(imgs, dim=0).to(device)

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

        traj_prior_imgs = test_imgs
        traj_prior_imgs_clip = test_imgs_clip

        # ---------------------------------- first pass ----------------------------------
        T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
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
            
            # 添加 source_view_mask: 前8张noisy图像对应1，剩余对应0
            total_images = len(curr_imgs)
            source_view_mask = torch.zeros(total_images, dtype=torch.float32, device=device)
            source_view_mask[:num_noisy] = 1
            value_dict["source_view_mask"] = source_view_mask[..., None, None, None]
            
            # 确保所有tensor都在正确的设备上
            for key, value in value_dict.items():
                if isinstance(value, torch.Tensor):
                    value_dict[key] = value.to(device)
            
            return value_dict


    def forward(self, x_target, x_target_latent, x_source_latent, batch):
        '''
        x_target: input rgb (bs*f, 3, h, w) all visible views
        x_target_latent: input latent (bs*f, 5, h, w) target latent
        x_source_latent: source latent (bs*f, 5, h, w) input latent of source image
        batch: Dict, used to obtain conditioning
        return: loss, loss_dict
        '''
        bs, num_f = x_target_latent.shape[0]//self.VERSION_DICT['T'], self.VERSION_DICT['T']
        c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)
        additional_model_inputs['source_view_mask'] = batch['source_view_mask']

        
        c = uc if self.training and random.random() < self.ucg_rate else c # unconditional guidance
        c['source_latent'] = x_source_latent
        sigmas = self.sigma_sampler(bs).unsqueeze(1).repeat(1, num_f).flatten().to(x_target_latent)
        print(f'at rank {int(os.environ["LOCAL_RANK"])}, sigmas: {sigmas[::21]}, target_latent_max:{x_target_latent.max()}')

        noise = torch.randn_like(x_target_latent)
        
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (x_target_latent.shape[0], 1, x_target_latent.shape[2])
                if self.n_frames is not None
                else (x_target_latent.shape[0], x_target_latent.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=x_target_latent.device),
                x_target_latent.ndim,
            )
        sigmas_bc = append_dims(sigmas, x_target_latent.ndim)
        # todo: liyi
        noised_input = self.get_noised_input(sigmas_bc, noise, x_target_latent)
        # noised_input = self.get_noised_input(sigmas_bc, noise, torch.randn_like(x_target_latent))
        model_output = self.DENOISER(
            self.MODEL, noised_input, sigmas, c, **additional_model_inputs
        )

        # original generative loss  
        w = append_dims(self.loss_weighting(sigmas), x_target_latent.ndim)
        w[additional_sampler_inputs['input_frame_mask']] = 0
        loss = self.get_loss(model_output, x_target_latent, w)
        if int(os.environ['LOCAL_RANK']) == -1:
            print(f'sigma: {sigmas.view(bs, num_f).mean(dim=1)}')
            unint_weight = torch.ones_like(w)
            unint_weight[additional_sampler_inputs['input_frame_mask']] = 0
            mse = self.get_loss(model_output, x_target_latent, unint_weight).view(bs, num_f).mean(dim=1)
            # x_source_latent_expand = model_output.clone()
            # x_source_latent_expand[~batch['cond_frames_mask']] = x_source_latent
            source_gt_mse = self.get_loss(x_source_latent, x_target_latent, unint_weight).view(bs, num_f).mean(dim=1)
            print(f'MSE: {mse}')
            print(f'source_gt_mse: {source_gt_mse}')
            print(f'loss: {loss.view(bs, num_f).mean(dim=1)}')
            with torch.no_grad():
                for bs_id in range(bs):
                    model_output_rgb = torch.utils.checkpoint.checkpoint(self.SEVA_CONDITIONER.ae.decode, model_output[bs_id*num_f:(bs_id+1)*num_f], self.options["decoding_t"])
                    source_rgb = torch.utils.checkpoint.checkpoint(self.SEVA_CONDITIONER.ae.decode, x_source_latent[bs_id*num_f:(bs_id+1)*num_f], self.options["decoding_t"])
                    save_vis = torch.cat((source_rgb/2+0.5, model_output_rgb/2+0.5, x_target[bs_id*num_f:(bs_id+1)*num_f]/2+0.5),dim=2)
                    save_image(save_vis, f'source_pred_gt_bs{bs_id}_sigma{sigmas[bs_id*num_f]}_mse{mse[bs_id]}.png')
                    pass
        loss_mean = loss.mean()


        # ######################################
        # FROZEN_MODEL = SGMWrapper_baseline(load_model_baseline(device="cpu", verbose=True)).to(x_target_latent)
        # with torch.inference_mode():
        #     for i in range(0, len(self.sigma_sampler.sigmas), 49):
        #         _sigmas = (torch.ones_like(sigmas) * self.sigma_sampler.sigmas[i]).to(sigmas)
        #         _sigmas_bc = append_dims(_sigmas.to(x_target_latent), x_target_latent.ndim)
        #         _noised_input = self.get_noised_input(_sigmas_bc, noise, x_target_latent)
        #         _model_output = self.DENOISER(
        #             self.MODEL, _noised_input, _sigmas, c, **additional_model_inputs
        #         )       
        #         # _model_output = self.DENOISER(
        #             # FROZEN_MODEL, _noised_input, _sigmas, c, **additional_model_inputs
        #         # )       
        #         _w = append_dims(self.loss_weighting(_sigmas), x_target_latent.ndim)
        #         _w[additional_sampler_inputs['input_frame_mask']] = 0
        #         _loss = self.get_loss(_model_output, x_target_latent, _w)
        #         print(f'when sigma: {_sigmas.mean()}, loss: {_loss.mean()}')
        #########################################
        return loss_mean
    
        # original generative loss  
        w = append_dims(self.loss_weighting(sigmas), x.ndim)
        loss = self.get_loss(model_output, x, w)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input
    
    def shared_step(self, batch: Dict) -> Any:
        # with torch.inference_mode():
        for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "edited_imgs", "source_view_mask"]:
            batch[key] = batch[key].flatten(start_dim=0, end_dim=1).contiguous()
        x_target = self.get_input(batch)
        x_source = batch['edited_imgs']
        x_target_latent = self.encode_first_stage(x_target)
        x_source_latent = self.encode_first_stage(x_source)
        
        batch["global_step"] = self.global_step
        loss = self(x_target, x_target_latent, x_source_latent, batch)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        print('validation step')
        return 0

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.logger.log_metrics({'loss': loss,
                                'global_step': self.global_step,
                                'lr': self.optimizers().param_groups[0]["lr"] if self.scheduler_config is not None else 0,
                                })
        
        # self.log_dict(
        #     {'loss': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=False
        # )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
    
        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
        return loss

    def _select_closest_cameras(self, available_perfect_indices, available_generated_indices, target_indices, all_c2ws, all_Ks):
        """
        分别从完美帧和生成帧中选择条件帧
        Args:
            available_perfect_indices: 可用的完美帧索引列表 [0,1,2,3,4,5]
            available_generated_indices: 可用的生成帧索引列表
            target_indices: 当前批次的目标索引列表 (8个)
            all_c2ws: 所有帧的外参 [num_frames, 4, 4]
            all_Ks: 所有帧的内参 [num_frames, 3, 3]
        Returns:
            selected_indices: 选择的5个条件帧索引
        """
        target_c2ws = all_c2ws[target_indices]  # [8, 4, 4]
        
        # 定义关键帧：首帧(0)、中间帧(3-4)、尾帧(7)
        first_frame_c2w = target_c2ws[0:1]  # [1, 4, 4]
        middle_frame_c2w = target_c2ws[3:5]  # [2, 4, 4] - 取中间2帧求平均
        last_frame_c2w = target_c2ws[7:8]   # [1, 4, 4]
        
        selected_indices = []
        
        # === 第一步：从available_perfect_indices中选择3个帧 ===
        if len(available_perfect_indices) > 0:
            perfect_c2ws = all_c2ws[available_perfect_indices]  # [N_perfect, 4, 4]
            
            # 计算距离
            perfect_first_rotation_dist = get_camera_dist(first_frame_c2w, perfect_c2ws, mode="rotation")[0]  # [N_perfect]
            perfect_first_translation_dist = get_camera_dist(first_frame_c2w, perfect_c2ws, mode="translation")[0]  # [N_perfect]
            perfect_first_combined_dist = perfect_first_rotation_dist + perfect_first_translation_dist * 100
            
            perfect_middle_rotation_dist = get_camera_dist(middle_frame_c2w, perfect_c2ws, mode="rotation").mean(0)  # [N_perfect]
            perfect_middle_translation_dist = get_camera_dist(middle_frame_c2w, perfect_c2ws, mode="translation").mean(0)  # [N_perfect]
            perfect_middle_combined_dist = perfect_middle_rotation_dist + perfect_middle_translation_dist * 100
            
            perfect_last_rotation_dist = get_camera_dist(last_frame_c2w, perfect_c2ws, mode="rotation")[0]  # [N_perfect]
            perfect_last_translation_dist = get_camera_dist(last_frame_c2w, perfect_c2ws, mode="translation")[0]  # [N_perfect]
            perfect_last_combined_dist = perfect_last_rotation_dist + perfect_last_translation_dist * 100
            
            print(f"=== Perfect Frames Selection ===")
            print(f"Perfect indices: {available_perfect_indices}")
            print(f"First frame distances: {perfect_first_combined_dist.cpu().numpy()}")
            print(f"Middle frame distances: {perfect_middle_combined_dist.cpu().numpy()}")
            print(f"Last frame distances: {perfect_last_combined_dist.cpu().numpy()}")
            
            used_perfect_mask = torch.zeros(len(available_perfect_indices), dtype=torch.bool)
            
            # 选择距离首帧最近的1个
            first_sorted_indices = torch.argsort(perfect_first_combined_dist)
            for idx in first_sorted_indices:
                if not used_perfect_mask[idx]:
                    selected_indices.append(available_perfect_indices[idx.item()])
                    used_perfect_mask[idx] = True
                    print(f"Selected from perfect (first): frame_{available_perfect_indices[idx.item()]} (dist: {perfect_first_combined_dist[idx].item():.4f})")
                    break
            
            # 选择距离中间帧最近的1个
            middle_sorted_indices = torch.argsort(perfect_middle_combined_dist)
            for idx in middle_sorted_indices:
                if not used_perfect_mask[idx]:
                    selected_indices.append(available_perfect_indices[idx.item()])
                    used_perfect_mask[idx] = True
                    print(f"Selected from perfect (middle): frame_{available_perfect_indices[idx.item()]} (dist: {perfect_middle_combined_dist[idx].item():.4f})")
                    break
            
            # 选择距离尾帧最近的1个
            last_sorted_indices = torch.argsort(perfect_last_combined_dist)
            for idx in last_sorted_indices:
                if not used_perfect_mask[idx]:
                    selected_indices.append(available_perfect_indices[idx.item()])
                    used_perfect_mask[idx] = True
                    print(f"Selected from perfect (last): frame_{available_perfect_indices[idx.item()]} (dist: {perfect_last_combined_dist[idx].item():.4f})")
                    break
        
        # === 第二步：从available_generated_indices中选择2个帧 ===
        if len(available_generated_indices) > 0:
            generated_c2ws = all_c2ws[available_generated_indices]  # [N_generated, 4, 4]
            
            # 计算距离
            generated_first_rotation_dist = get_camera_dist(first_frame_c2w, generated_c2ws, mode="rotation")[0]  # [N_generated]
            generated_first_translation_dist = get_camera_dist(first_frame_c2w, generated_c2ws, mode="translation")[0]  # [N_generated]
            generated_first_combined_dist = generated_first_rotation_dist + generated_first_translation_dist * 100
            
            generated_last_rotation_dist = get_camera_dist(last_frame_c2w, generated_c2ws, mode="rotation")[0]  # [N_generated]
            generated_last_translation_dist = get_camera_dist(last_frame_c2w, generated_c2ws, mode="translation")[0]  # [N_generated]
            generated_last_combined_dist = generated_last_rotation_dist + generated_last_translation_dist * 100
            
            print(f"=== Generated Frames Selection ===")
            print(f"Generated indices: {available_generated_indices}")
            print(f"First frame distances: {generated_first_combined_dist.cpu().numpy()}")
            print(f"Last frame distances: {generated_last_combined_dist.cpu().numpy()}")
            
            used_generated_mask = torch.zeros(len(available_generated_indices), dtype=torch.bool)
            
            # 选择距离首帧最近的1个
            first_sorted_indices = torch.argsort(generated_first_combined_dist)
            for idx in first_sorted_indices:
                if not used_generated_mask[idx]:
                    selected_indices.append(available_generated_indices[idx.item()])
                    used_generated_mask[idx] = True
                    print(f"Selected from generated (first): frame_{available_generated_indices[idx.item()]} (dist: {generated_first_combined_dist[idx].item():.4f})")
                    break
            
            # 选择距离尾帧最近的1个
            last_sorted_indices = torch.argsort(generated_last_combined_dist)
            for idx in last_sorted_indices:
                if not used_generated_mask[idx]:
                    selected_indices.append(available_generated_indices[idx.item()])
                    used_generated_mask[idx] = True
                    print(f"Selected from generated (last): frame_{available_generated_indices[idx.item()]} (dist: {generated_last_combined_dist[idx].item():.4f})")
                    break
        
        # === 第三步：如果不足5个，从available_perfect_indices中添加还没被选中的帧 ===
        while len(selected_indices) < 5:
            if len(available_perfect_indices) > 0:
                # 找出还没有被选中的完美帧
                unused_perfect = [idx for idx in available_perfect_indices if idx not in selected_indices]
                if unused_perfect:
                    # 优先选择还没有被使用过的完美帧
                    additional_idx = unused_perfect[0]  # 简单取第一个，也可以基于距离选择
                    selected_indices.append(additional_idx)
                    print(f"Added unused perfect frame to meet 5-frame requirement: frame_{additional_idx}")
                else:
                    # 如果所有完美帧都已经被选中，说明完美帧数量不足
                    print(f"WARNING: All perfect frames already selected, but still need more frames. Available perfect: {available_perfect_indices}")
                    break
            else:
                # 如果连完美帧都没有，出现异常
                print("ERROR: No perfect frames available!")
                break
        
        print(f"=== Final Selection Summary ===")
        print(f"Selected {len(selected_indices)} condition frames: {selected_indices}")
        perfect_count = len([idx for idx in selected_indices if idx in available_perfect_indices])
        generated_count = len([idx for idx in selected_indices if idx in available_generated_indices])
        print(f"From perfect frames: {perfect_count} frames")
        print(f"From generated frames: {generated_count} frames")
        print(f"Perfect frames used: {[idx for idx in selected_indices if idx in available_perfect_indices]}")
        print(f"Generated frames used: {[idx for idx in selected_indices if idx in available_generated_indices]}")
        print(f"=== End Selection Summary ===")
        
        return selected_indices

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        '''
        基于相机距离的动态条件帧选择策略
        batch:
        {
            'noisy_imgs': torch.Tensor, shape: [num_video_frames, H, W, 3], range: [0, 255]
            'scene_name': str
            'img_names': list[str], shape: [num_video_frames]
            'noisy_img_intrinsics': torch.Tensor, shape: [num_video_frames, 3, 3]
            'noisy_img_extrinsics': torch.Tensor, shape: [num_video_frames, 4, 4]
        }
        '''
        import math
        import os
        assert len(batch['noisy_imgs']) == 1, "Batch size should be 1 for testing"
        
        # 获取完整视频数据
        noisy_imgs = batch['noisy_imgs'][0]  # [num_frames, H, W, 3], range [0, 255]
        scene_name = batch['scene_name'][0]
        img_names = [item[0] for item in batch['img_names']]
        noisy_intrinsics = batch['noisy_img_intrinsics'][0]  # [num_frames, 3, 3]
        noisy_extrinsics = batch['noisy_img_extrinsics'][0]  # [num_frames, 4, 4]
        
        num_frames = len(noisy_imgs)
        
        # 从test_data_config中获取datafolder的文件夹名
        datafolder_name = "hunyuan_test_select_0918"  # 默认值
        if self.test_data_config:
            try:
                if isinstance(self.test_data_config, dict):
                    datafolder = self.test_data_config.get('params', {}).get('datafolder', '')
                else:  # OmegaConf or similar
                    datafolder = getattr(self.test_data_config, 'params', {}).get('datafolder', '') if hasattr(self.test_data_config, 'params') else ''
                
                if datafolder:
                    datafolder_name = os.path.basename(datafolder.rstrip('/'))
            except Exception as e:
                print(f"Warning: Could not extract datafolder from test_data_config: {e}")
        
        save_dir = os.path.join(self.save_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing scene: {scene_name} with {num_frames} frames")
        print(f"=== Dynamic Camera-Distance-Based Condition Selection ===")
        
        # 存储所有去噪结果
        denoised_frames = {}  # {frame_idx: tensor}
        
        # 初始化两个不同的索引列表
        available_perfect_indices = [0, 1, 2, 3, 4, 5]  # 完美的初始帧
        available_generated_indices = []  # 生成的帧
        
        # 保存初始可用帧到目标文件夹
        for idx in available_perfect_indices:
            if idx < num_frames:
                img = noisy_imgs[idx]  # [H, W, 3], range [0, 255]
                img_pil = Image.fromarray(img.cpu().numpy().astype(np.uint8))
                img_name = os.path.splitext(img_names[idx])[0]  # 去掉扩展名
                img_pil.save(os.path.join(save_dir, f"{img_name}.png"))
                denoised_frames[idx] = img.permute(2, 0, 1).float() / 255.0  # 保存为[3, H, W], range [0, 1]
                print(f"Saved initial available frame {idx}: {img_name}.png")
        
        # 需要去噪的目标帧（从第6帧开始）
        remaining_indices = [i for i in range(6, num_frames)]
        
        print(f"Initial perfect frames: {available_perfect_indices}")
        print(f"Initial generated frames: {available_generated_indices}")
        print(f"Frames to denoise: {remaining_indices}")
        
        # 批量处理：每批8个目标帧
        batch_size = 8
        for batch_num, i in enumerate(range(0, len(remaining_indices), batch_size)):
            batch_target_indices = remaining_indices[i:i+batch_size]
            original_batch_target_indices = batch_target_indices.copy()  # 保存原始索引
            
            # 如果不足8个，复制最后一个索引填充
            if len(batch_target_indices) < batch_size:
                last_index = batch_target_indices[-1]
                padded_count = batch_size - len(batch_target_indices)
                batch_target_indices = batch_target_indices + [last_index] * padded_count
                print(f"\n--- Processing batch {batch_num}: PADDED BATCH ---")
                print(f"Original target indices to denoise: {original_batch_target_indices}")
                print(f"Padded target indices (for processing): {batch_target_indices}")
                print(f"Padded {padded_count} frames by repeating frame {last_index}")
            else:
                print(f"\n--- Processing batch {batch_num}: FULL BATCH ---")
                print(f"Target indices to denoise: {batch_target_indices}")
            
            print(f"Current perfect frames for condition selection: {available_perfect_indices}")
            print(f"Current generated frames for condition selection: {available_generated_indices}")
            print(f"Batch size: {len(batch_target_indices)} (original: {len(original_batch_target_indices)})")
            
            # 使用相机距离选择5个最近的条件帧
            selected_cond_indices = self._select_closest_cameras(
                available_perfect_indices, 
                available_generated_indices,
                batch_target_indices, 
                noisy_extrinsics, 
                noisy_intrinsics
            )
            
            print(f"Selected condition frames: {selected_cond_indices}")
            print(f"Target frames to denoise: {batch_target_indices}")
            perfect_cond_count = len([idx for idx in selected_cond_indices if idx in available_perfect_indices])
            generated_cond_count = len([idx for idx in selected_cond_indices if idx in available_generated_indices])
            print(f"Condition frame types: Perfect={perfect_cond_count}, Generated={generated_cond_count}")
            print(f"Perfect condition frames: {[idx for idx in selected_cond_indices if idx in available_perfect_indices]}")
            print(f"Generated condition frames: {[idx for idx in selected_cond_indices if idx in available_generated_indices]}")
            
            # 准备条件帧
            cond_imgs_list = []
            cond_Ks_list = []
            cond_c2ws_list = []
            
            for idx in selected_cond_indices:
                if idx in denoised_frames:
                    # 使用已去噪的结果（值域[0,1] -> [0,255]）
                    cond_img = denoised_frames[idx] * 255.0  # [3, H, W], [0, 255]
                else:
                    # 使用原始图像
                    cond_img = noisy_imgs[idx].permute(2, 0, 1).float()  # [3, H, W], [0, 255]
                
                cond_imgs_list.append(cond_img)
                cond_Ks_list.append(noisy_intrinsics[idx])
                cond_c2ws_list.append(noisy_extrinsics[idx])
            
            # 准备目标帧
            noisy_imgs_batch = []
            noisy_Ks_batch = []
            noisy_c2ws_batch = []
            
            for idx in batch_target_indices:
                if idx < num_frames:
                    noisy_imgs_batch.append(noisy_imgs[idx].permute(2, 0, 1).float())  # [3, H, W], [0, 255]
                    noisy_Ks_batch.append(noisy_intrinsics[idx])
                    noisy_c2ws_batch.append(noisy_extrinsics[idx])
            
            # 转换为tensor
            cond_imgs_tensor = torch.stack(cond_imgs_list)  # [5, 3, H, W], [0, 255]
            cond_Ks_tensor = torch.stack(cond_Ks_list)      # [5, 3, 3]
            cond_c2ws_tensor = torch.stack(cond_c2ws_list)  # [5, 4, 4]
            
            noisy_imgs_tensor = torch.stack(noisy_imgs_batch)  # [8, 3, H, W], [0, 255]
            noisy_Ks_tensor = torch.stack(noisy_Ks_batch)      # [8, 3, 3]
            noisy_c2ws_tensor = torch.stack(noisy_c2ws_batch)  # [8, 4, 4]
            
            print(f"Condition frames shape: {cond_imgs_tensor.shape}")
            print(f"Target frames shape: {noisy_imgs_tensor.shape}")
            
            # 调用数据预处理函数
            processed_batch = self.noisy_imgs_Ks_c2ws_cond_imgs_K2_c2ws_2_batch(
                noisy_imgs_tensor,
                noisy_Ks_tensor,
                noisy_c2ws_tensor,
                cond_imgs_tensor,
                cond_Ks_tensor,
                cond_c2ws_tensor,
                self.VERSION_DICT
            )
            
            # 执行去噪推理
            batch_denoised = self._denoise_frames(processed_batch)
            
            # 保存结果并更新available_indices
            if processed_batch is not None and 'cond_frames_mask' in processed_batch:
                target_mask = processed_batch['cond_frames_mask'] == 0
                target_indices_in_batch = torch.where(target_mask)[0].cpu().numpy()
                
                batch_saved_indices = []
                for j, batch_idx_in_result in enumerate(target_indices_in_batch):
                    if j < len(original_batch_target_indices):  # 只处理原始索引，不处理填充的
                        original_idx = original_batch_target_indices[j]  # 使用原始索引
                        if original_idx < num_frames:  # 只处理有效索引
                            denoised_img = batch_denoised[batch_idx_in_result]  # [3, H, W], range [0, 1]
                            
                            # 保存去噪后的输出图像
                            img_name = img_names[original_idx]
                            save_path = os.path.join(save_dir, img_name)
                            self._save_image(denoised_img, save_path)
                            
                            # 保存对应的输入(noisy)图像用于对比
                            img_name_base = os.path.splitext(img_name)[0]
                            img_name_ext = os.path.splitext(img_name)[1] or ".png"
                            input_save_path = os.path.join(save_dir, f"{img_name_base}_input{img_name_ext}")
                            input_img = noisy_imgs[original_idx].permute(2, 0, 1).float() / 255.0  # [3, H, W], [0, 1]
                            self._save_image(input_img, input_save_path)
                            
                            # 存储结果
                            denoised_frames[original_idx] = denoised_img
                            batch_saved_indices.append(original_idx)
                            
                            print(f"Saved denoised frame {original_idx}: {img_name}")
                            print(f"Saved input frame {original_idx}: {img_name_base}_input{img_name_ext}")
                
                print(f"=== Batch {batch_num} Denoising Results ===")
                print(f"Original target indices for this batch: {original_batch_target_indices}")
                print(f"Successfully denoised indices: {batch_saved_indices}")
                print(f"Total frames denoised in this batch: {len(batch_saved_indices)}")
                print(f"Condition frames used: {selected_cond_indices}")
                
                # 将本批次成功去噪的帧加入available_generated_indices
                for idx in batch_saved_indices:
                    if idx not in available_generated_indices:
                        available_generated_indices.append(idx)
                
                print(f"Updated available_generated_indices after batch {batch_num}: {sorted(available_generated_indices)}")
                print(f"Total available frames count: perfect={len(available_perfect_indices)}, generated={len(available_generated_indices)}")
                print(f"--- End of batch {batch_num} processing ---\n")
        
        print(f"\n=== Final Denoising Summary ===")
        print(f"Total frames in video: {num_frames}")
        print(f"Initial perfect frames: {len(available_perfect_indices)} (indices {available_perfect_indices})")
        print(f"Frames processed for denoising: {len(denoised_frames) - len(available_perfect_indices)}")  # 减去初始帧
        print(f"Final perfect frames: {sorted(available_perfect_indices)}")
        print(f"Final generated frames: {sorted(available_generated_indices)}")
        print(f"Total available frames: perfect={len(available_perfect_indices)}, generated={len(available_generated_indices)}")
        print(f"Total denoised frames: {len(denoised_frames)}")
        print(f"=== Denoising completed successfully ===")
        
        # 按帧索引顺序将所有去噪图像拼接成视频并保存
        try:
            import imageio
            video_fps = self.VERSION_DICT.get("options", {}).get("video_save_fps", 30.0)
            sorted_indices = sorted(denoised_frames.keys())
            video_frames = []
            for idx in sorted_indices:
                frame = denoised_frames[idx]  # [3, H, W], range [0, 1]
                frame_np = (frame.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)  # [H, W, 3]
                video_frames.append(frame_np)
            
            video_path = os.path.join(save_dir, "denoised.mp4")
            imageio.mimwrite(video_path, video_frames, fps=video_fps, quality=8)
            print(f"Saved denoised video ({len(video_frames)} frames, {video_fps} fps): {video_path}")
        except Exception as e:
            print(f"Warning: Failed to save video: {e}")
        
        return {"denoised_frames": len(denoised_frames), "available_perfect_frames": len(available_perfect_indices), "available_generated_frames": len(available_generated_indices)}
    
    def _denoise_frames(self, batch):
        """
        执行帧去噪
        """
        import math
        
        # 设置全局步骤
        batch["global_step"] = self.global_step
        
        # 获取条件信息
        c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)
        
        # 准备采样形状
        num_samples = [batch['cond_frames'].shape[0] // self.VERSION_DICT["T"], self.VERSION_DICT["T"]]
        shape = (
            math.prod(num_samples), 
            self.VERSION_DICT["C"], 
            self.VERSION_DICT["H"] // self.VERSION_DICT["f"], 
            self.VERSION_DICT["W"] // self.VERSION_DICT["f"]
        )
        
        # 设置无条件引导
        uc = dict()
        for key in c:
            uc[key] = c[key].clone()
        
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            # 生成随机噪声
            randn = torch.randn(shape).to(self.device)
            
            # 执行采样
            samples_z = self.sampler(
                lambda input, sigma, c: self.DENOISER(
                    self.MODEL,
                    input,
                    sigma,
                    c,
                    **additional_model_inputs,
                    source_view_mask=torch.cat([
                        batch['source_view_mask'], 
                        torch.ones_like(batch['source_view_mask'])
                    ], dim=0)
                ),
                randn,
                scale=(
                    self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) 
                    else self.options["cfg"]
                ),
                cond=c,
                uc=uc,
                verbose=None,
                return_intermediate=False,
                **additional_sampler_inputs,
            )
            
            # 解码为RGB图像
            import deepspeed
            samples = deepspeed.checkpointing.checkpoint(
                self.SEVA_CONDITIONER.ae.decode, 
                samples_z.to(self.device), 
                self.options["decoding_t"]
            )
            
            # 如果返回的是tuple，取第一个元素
            if isinstance(samples, tuple):
                samples = samples[0]
            
            # 转换范围从[-1, 1]到[0, 1]并限制范围
            samples = samples.clamp(-1, 1) / 2 + 0.5
            
            return samples
    
    def _save_image(self, img_tensor, save_path):
        """
        保存图像tensor到文件
        Args:
            img_tensor: [3, H, W], range [0, 1]
            save_path: 保存路径
        """
        from torchvision.utils import save_image
        save_image(img_tensor, save_path)
        print(f"Saved image: {save_path}")

    def _check_images_exist(self, batch):
        """
        检查所有要保存的图像是否已经存在
        返回 True 如果所有图像都存在，False 否则
        """
        save_dir = self.save_dir
        
        # 获取batch信息
        batch_size = batch['cond_frames'].shape[0] // self.VERSION_DICT["T"]
        T = self.VERSION_DICT["T"]
        
        for b in range(batch_size):
            batch_start = b * T
            batch_end = (b + 1) * T
            
            # 获取当前batch的数据
            source_view_mask = batch['source_view_mask'][batch_start:batch_end, 0, 0, 0]  # [T]
            cond_frames_mask = batch['cond_frames_mask'][batch_start:batch_end]  # [T]
            
            # 获取场景名称
            if 'name' in batch:
                scene_name = batch['name'][b] if isinstance(batch['name'], list) else f"scene_{b}"
            else:
                scene_name = f"batch_{batch_start//T}"
            
            # 检查source images是否存在
            source_indices = torch.where(source_view_mask == 1)[0].cpu().numpy()
            for idx in source_indices:
                filename = f"{scene_name}_{idx}.png"
                filepath = os.path.join(save_dir, filename)
                if not os.path.exists(filepath):
                    return False
            
            # 检查生成的target images是否存在
            target_indices = torch.where(cond_frames_mask == 0)[0].cpu().numpy()
            for source_idx, idx in enumerate(target_indices):
                filename = f"{scene_name}_{source_idx}_seva_denoised.png"
                filepath = os.path.join(save_dir, filename)
                if not os.path.exists(filepath):
                    return False
        
        return True

    def _save_comparison_images(self, batch, log):
        """
        保存对比图像：source_view_mask==1的图像和cond_frames_mask==0的图像分别保存
        """
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取batch信息
        batch_size = batch['cond_frames'].shape[0] // self.VERSION_DICT["T"]
        T = self.VERSION_DICT["T"]
        
        for b in range(batch_size):
            batch_start = b * T
            batch_end = (b + 1) * T
            
            # 获取当前batch的数据
            source_view_mask = batch['source_view_mask'][batch_start:batch_end, 0, 0, 0]  # [T]
            cond_frames_mask = batch['cond_frames_mask'][batch_start:batch_end]  # [T]
            cond_frames = batch['cond_frames'][batch_start:batch_end]  # [T, 3, H, W]
            samples = log["samples"][batch_start:batch_end]  # [T, 3, H, W]
            
            # 获取场景名称
            if 'name' in batch:
                scene_name = batch['name'][b] if isinstance(batch['name'], list) else f"scene_{b}"
            else:
                scene_name = f"batch_{batch_start//T}"
            
            # 保存source images (noisy input images)
            source_indices = torch.where(source_view_mask == 1)[0].cpu().numpy()
            for idx in source_indices:
                img = cond_frames[idx]  # [3, H, W], range [-1, 1]
                img = (img + 1) / 2  # 转换到[0, 1]
                img = img.clamp(0, 1)
                
                filename = f"{scene_name}_{idx}.png"
                filepath = os.path.join(save_dir, filename)
                save_image(img, filepath)
                print(f"Saved source image: {filepath}")
            
            # 保存生成的target images 
            target_indices = torch.where(cond_frames_mask == 0)[0].cpu().numpy()
            for source_idx, idx in enumerate(target_indices):
                img = samples[idx]  # [3, H, W], range [0, 1]
                img = img.clamp(0, 1)
                
                # 使用对应source image的文件名加上_seva_denoised后缀
                filename = f"{scene_name}_{source_idx}_seva_denoised.png"
                filepath = os.path.join(save_dir, filename)
                save_image(img, filepath)
                print(f"Saved generated image: {filepath}")
            
            print(f"Batch {b}: Saved {len(source_indices)} source images and {len(target_indices)} generated images")
            print(f"  - Scene: {scene_name}")
            print(f"  - Source indices (noisy): {source_indices}")
            print(f"  - Target indices (generated): {target_indices}")

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.MODEL.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log


    @rank_zero_only
    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:  
        

        # # single step v2
        # log = dict()
        # for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "edited_imgs"]:
        #     batch[key] = batch[key].flatten(start_dim=0, end_dim=1)
        # x_target = self.get_input(batch)
        # x_source = batch['edited_imgs']
        # x_target_latent = self.encode_first_stage(x_target)
        # x_source_latent = self.encode_first_stage(x_source)

        # log["inputs"] = x_source
        # log["target"] = x_target
        # z0 = self.SEVA_CONDITIONER.ae.encode(x_source, self.options["encoding_t"])
        # log["reconstructions"] = self.SEVA_CONDITIONER.ae.decode(z0, self.options["decoding_t"])
        # batch["global_step"] = self.global_step
        # c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)
        # c = uc if self.training and random.random() < self.ucg_rate else c # unconditional guidance
        # # sigmas = self.sigma_sampler(x_target_latent.shape[0]).to(x_target_latent)
        # # noise = torch.randn_like(x_target_latent)
        # sigmas = self.sampler.get_max_sigma(x_target_latent.shape[0]).to(x_target_latent)
        # noise = x_source_latent
        # if self.offset_noise_level > 0.0:
        #     offset_shape = (
        #         (x_target_latent.shape[0], 1, x_target_latent.shape[2])
        #         if self.n_frames is not None
        #         else (x_target_latent.shape[0], x_target_latent.shape[1])
        #     )
        #     noise = noise + self.offset_noise_level * append_dims(
        #         torch.randn(offset_shape, device=x_target_latent.device),
        #         x_target_latent.ndim,
        #     )
            
        # sigmas_bc = append_dims(sigmas, x_target_latent.ndim)
        # # noised_input = self.get_noised_input(sigmas_bc, noise, x_target_latent)
        # noised_input = self.get_noised_input(sigmas_bc, noise, x_source_latent)


        # model_output = self.DENOISER(
        #     self.MODEL, noised_input, sigmas, c, **additional_model_inputs
        # )

        # samples_singlepass = self.SEVA_CONDITIONER.ae.decode(model_output, self.options["decoding_t"])
        # log["samples"] = samples_singlepass
        # del x_source_latent, model_output
        # torch.cuda.empty_cache()
        # return log





        # # single step v1
        # log = dict()
        # for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "edited_imgs"]:
        #     batch[key] = batch[key].flatten(start_dim=0, end_dim=1)
        # x_target = self.get_input(batch)
        # x_source = batch['edited_imgs']
        # x_target_latent = self.encode_first_stage(x_target)
        # x_source_latent = self.encode_first_stage(x_source)

        # log["inputs"] = x_source
        # z0 = self.SEVA_CONDITIONER.ae.encode(x_source, self.options["encoding_t"])
        # log["reconstructions"] = self.SEVA_CONDITIONER.ae.decode(z0, self.options["decoding_t"])
        # batch["global_step"] = self.global_step
        # c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)
        # num_samples = [x_source.shape[0]//self.VERSION_DICT["T"], self.VERSION_DICT["T"]]
        # shape = (math.prod(num_samples), self.VERSION_DICT["C"], self.VERSION_DICT["H"] // self.VERSION_DICT["f"], self.VERSION_DICT["W"] // self.VERSION_DICT["f"])
        # # 1. generation from randn noise
        # with torch.inference_mode(), torch.autocast("cuda"):
        #     # randn = torch.randn(shape).to("cuda")
        #     randn = x_source_latent.to("cuda")
        #     samples_z = self.sampler.single_step(
        #                 lambda input, sigma, c: self.DENOISER(
        #                     self.MODEL,
        #                     input,
        #                     sigma,
        #                     c,
        #                     **additional_model_inputs,
        #                 ),
        #                 randn,
        #                 scale=(
        #                     self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]
        #                 ),
        #                 cond=c,
        #                 uc=uc,
        #                 verbose=None,
        #                 **additional_sampler_inputs,
        #             )
        # samples_singlepass = self.SEVA_CONDITIONER.ae.decode(samples_z, self.options["decoding_t"])
        # log["samples"] = samples_singlepass
        # del randn, samples_z
        # torch.cuda.empty_cache()
        # return log
    



        # multi step 
        log = dict()    
        bs, num_f = 1, self.VERSION_DICT['T']
        for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "edited_imgs", "source_view_mask"]:
            batch[key] = batch[key][0:bs].flatten(start_dim=0, end_dim=1).contiguous()
        x = self.get_input(batch)
        log["target"] = x
        x_source = batch['edited_imgs']
        log["inputs"] = x_source
        x_source_latent = self.encode_first_stage(x_source).to(x_source)
        x_target_latent = self.SEVA_CONDITIONER.ae.encode(x, self.options["encoding_t"])
        log["reconstructions"] = self.SEVA_CONDITIONER.ae.decode(x_target_latent, self.options["decoding_t"])
        batch["global_step"] = self.global_step
        c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)

        num_samples = [x.shape[0]//self.VERSION_DICT["T"], self.VERSION_DICT["T"]]
        shape = (math.prod(num_samples), self.VERSION_DICT["C"], self.VERSION_DICT["H"] // self.VERSION_DICT["f"], self.VERSION_DICT["W"] // self.VERSION_DICT["f"])
        randn = torch.randn(shape).to("cuda")


        # noisy input
        # ========================= set the sourceview 5-th channel to -1 to distinguish from the cond view and target view ======================
        # c['concat'] = self.update_c_concat(c['concat']
        # , bs, num_f, (~batch['cond_frames_mask']).sum(), sigmas)
        # ========================= set the sourceview 5-th channel to -1 to distinguish from the cond view and target view ======================      

        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # ====================== Test different sigma values for denoising performance ======================
                # Test sigma values: sample from different levels of the sigma schedule
                test_sigma_indices = [0, 200, 400, 600, 800, 999]  # Different levels
                sigma_comparison_metrics = {}
                
                print("Starting sigma comparison test...")
                sigmas = self.sigma_sampler(bs).unsqueeze(1).repeat(1, num_f).flatten().to(x_source_latent)
                noise = torch.randn_like(x_source_latent)     
                w = append_dims(self.loss_weighting(sigmas), x_source_latent.ndim)
                unint_weight = torch.ones_like(w)
                unint_weight[additional_sampler_inputs['input_frame_mask']] = 0
                for sigma_idx in test_sigma_indices:
                    if sigma_idx < len(self.sigma_sampler.sigmas):
                        test_sigma_value = self.sigma_sampler.sigmas[sigma_idx]
                        test_sigmas = (torch.ones_like(sigmas) * test_sigma_value).to(sigmas)
                        test_sigmas_bc = append_dims(test_sigmas, x_source_latent.ndim)
                        
                        # Create noised input for this sigma level
                        test_noised_input = self.get_noised_input(test_sigmas_bc, noise, x_target_latent)
                        
                        # Test baseline model (need to reuse FROZEN_MODEL or create temporarily)
                        baseline_output = self.BASELINE_DENOISER(
                            FROZEN_MODEL,
                            test_noised_input.clone(),
                            test_sigmas,
                            c,
                            **additional_model_inputs
                        )
                        baseline_loss = self.get_loss(baseline_output, x_target_latent, unint_weight).mean()
                        log[f"denoise_baseline_sigma_{test_sigma_value}"] = self.SEVA_CONDITIONER.ae.decode(baseline_output, self.options["decoding_t"])

                        # Test our model
                        our_output = self.DENOISER(
                            self.MODEL,
                            test_noised_input.clone(),
                            test_sigmas,
                            c,
                            **additional_model_inputs,
                            source_view_mask=batch['source_view_mask']
                        )
                        our_loss = self.get_loss(our_output, x_target_latent, unint_weight).mean()
                        log[f"denoise_ours_sigma_{test_sigma_value}"] = self.SEVA_CONDITIONER.ae.decode(our_output, self.options["decoding_t"])
                        
                        # Calculate advantage (negative means we're better)
                        advantage = our_loss - baseline_loss
                        sigma_comparison_metrics[f'sigma_advantage_{sigma_idx:02d}'] = advantage.item()
                        
                        print(f'Sigma {sigma_idx} (value={test_sigma_value:.4f}): Baseline Loss={baseline_loss:.6f}, Our Loss={our_loss:.6f}, Advantage={advantage:.6f}')

                # Log the sigma comparison metrics
                self.logger.log_metrics(sigma_comparison_metrics)
                print("Sigma comparison test completed.")
                # ====================== End sigma comparison test ======================


                # ==================== begin sampling test =======================
                samples_z_baseline, intermediate_z_baseline = self.sampler(
                        lambda input, sigma, c: self.BASELINE_DENOISER(
                            FROZEN_MODEL,
                            input,
                            sigma,
                            c,
                            **additional_model_inputs,
                        ),
                        randn.clone(),
                        scale=(
                            self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]
                        ),
                        cond=c,
                        uc=uc,
                        verbose=None,
                        return_intermediate=True,
                        **additional_sampler_inputs,
                    )
                # liyi todo: uc=c,
                uc = dict()
                for key in c:
                    uc[key] = c[key].clone()

                samples_z_ours, intermediate_z_ours = self.sampler(
                            lambda input, sigma, c: self.DENOISER(
                                self.MODEL,
                                input,
                                sigma,
                                c,
                                **additional_model_inputs,
                                source_view_mask=torch.cat([batch['source_view_mask'], torch.ones_like(batch['source_view_mask'])], dim=0)
                            ),
                            torch.randn_like(x_source_latent).to(x_source_latent),
                            scale=(
                                self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]
                            ),
                            cond=c,
                            uc=uc,
                            verbose=None,
                            return_intermediate=True,
                            **additional_sampler_inputs,
                        )
            
                samples_ours = torch.utils.checkpoint.checkpoint(self.SEVA_CONDITIONER.ae.decode, samples_z_ours, self.options["decoding_t"])
                log["samples"] = samples_ours
                for i in [0, 10, 30, 40, 48]:
                    log[f'samples_ours_step_{i}'] = torch.utils.checkpoint.checkpoint(self.SEVA_CONDITIONER.ae.decode, intermediate_z_ours[i], self.options["decoding_t"])
                # psnr = calculate_psnr_pt(samples.clamp(-1, 1)/2 + 0.5, x/2 + 0.5, 0).mean().cpu().numpy()
                # ssim = calculate_ssim_pt(samples.clamp(-1, 1)/2 + 0.5, x/2 + 0.5, 0).mean().cpu().numpy()
                self.logger.log_metrics({'advantage_step_00': ((intermediate_z_ours[0][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean()-((intermediate_z_baseline[0][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean(),
                                        'advantage_step_10': ((intermediate_z_ours[10][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean()-((intermediate_z_baseline[10][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean(),
                                        'advantage_step_30': ((intermediate_z_ours[30][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean()-((intermediate_z_baseline[30][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean(),
                                        'advantage_step_40': ((intermediate_z_ours[40][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean()-((intermediate_z_baseline[40][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean(),
                                        'advantage_step_48': ((intermediate_z_ours[48][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean()-((intermediate_z_baseline[48][~batch['cond_frames_mask']]-x_target_latent[~batch['cond_frames_mask']])**2).mean(),
                                        })

                # ==================== end sampling test =======================


                # Now clean up FROZEN_MODEL after all tests are complete
                del FROZEN_MODEL
                torch.cuda.empty_cache()
                del randn, samples_z_ours, intermediate_z_ours, samples_z_baseline, intermediate_z_baseline

        
        # add more detailed samples
        # for i in [0 ]:
        #     for scale in [1.2, 1.5, 2.0]:
        #         samples_z = self.sampler.add_noise_and_sample(
        #             lambda input, sigma, c: self.DENOISER(
        #                         self.MODEL,
        #                         input,
        #                         sigma,
        #                         c,
        #                         **additional_model_inputs,
        #                     ),
        #                     z0,
        #                     scale=(
        #                         # self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]
        #                         scale
        #                     ),
        #                     cond=c,
        #                     # uc=uc, # no uncond
        #                     uc=c, 
        #                     num_noise_steps=i,
        #                     verbose=None,
        #                     **additional_sampler_inputs,
        #                 )
        #         samples = self.SEVA_CONDITIONER.ae.decode(samples_z, self.options["decoding_t"])
        #         log[f"samples_edit_numstep={i}_scale={scale}"] = samples.clone()

        return log
    
        # 2. generation from prior frames
        for i in [10, 20, 30, 40]:
            with torch.inference_mode(), torch.autocast("cuda"):
                samples_z = self.sampler.add_noise_and_sample(
                            lambda input, sigma, c: self.DENOISER(
                                self.MODEL,
                                input,
                                sigma,
                                c,
                                **additional_model_inputs,
                            ),
                            z0,
                            scale=(
                                self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]
                            ),
                            cond=c,
                            uc=uc,
                            num_noise_steps=50-i,
                            verbose=None,
                            **additional_sampler_inputs,
                        )
            samples = self.SEVA_CONDITIONER.ae.decode(samples_z, self.options["decoding_t"])
            log[f"samples_sdedit_t=10_numstep={i}"] = samples

        del randn, samples_z, samples_singlepass
        torch.cuda.empty_cache()
        return log
















        ########## following code is source SDXL code, to be deleted ##########
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log



