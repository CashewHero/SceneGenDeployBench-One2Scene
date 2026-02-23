from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import Compose

import torch.nn.functional as F
from einops import rearrange

def read_list(list_file):
    rgb_list = []
    not_include_file = ['d7bcd8d5-755e-4de1-9782-7ae8d030a0d9', 'cb9a173c-859b-4ac1-9b30-3c099309c076', '39311111-417c-4c7f-b96e-8f0ae5f8a393', \
                        '91901e40-5f67-461a-aab9-110ab46970ea', '9201f6ab-4002-4a87-a9a6-3b56b7fbbc54', '451f9a32-8e68-46a0-9bb6-e60a3c2cf6a6', \
                        '3af3e0a7-4f80-4f7d-9f56-aad9381bc42c']
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            if not any(x in line for x in not_include_file):
                rgb_list.append(line.strip())
    return rgb_list

class Zind(data.Dataset):
    """The Zind Dataset with Label from Teacher Model"""

    def __init__(self, root_dir, list_file, height=504, width=1008, color_augmentation=True,
                 LR_filp_augmentation=True, yaw_rotation_augmentation=True, repeat=1, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Unlabeled Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 1.0
        self.min_depth_meters = 0.01

        self.color_augmentation = color_augmentation
        self.LR_filp_augmentation = LR_filp_augmentation
        self.yaw_rotation_augmentation = yaw_rotation_augmentation

        if self.color_augmentation:
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

        self.is_training = is_training

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        idx = (idx * 2) % len(self.rgb_list)
        # Read and process the image file
        rgb_name = self.rgb_list[idx]
        rgb = cv2.imread(rgb_name)
        if rgb is None:
            print(rgb_name)
            assert 0
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # Read and process the depth file
        depth_name = rgb_name.replace("/hpc2hdd/home/zcao740/Documents/Dataset", "/hpc2hdd/home/zcao740/Documents/360Depth/Semi-supervision/unlabel_depth") 
        depth_name = depth_name.replace("/hpc2hdd/home/zcao740/Downloads", "/hpc2hdd/home/zcao740/Documents/360Depth/Semi-supervision/unlabel_depth") 
        depth_name = depth_name.replace("pano_", "depth_").replace(".jpg", ".npy")
        try:
            gt_depth = np.load(depth_name)
        except:
            print(depth_name)
            assert 0
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float)
        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters
        
        raw_rgb = rgb.copy()
        if self.is_training and self.color_augmentation:
            strong_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(raw_rgb)))
        else:
            strong_rgb = raw_rgb.copy()
        raw_rgb = self.to_tensor(raw_rgb.copy())
        strong_rgb = self.to_tensor(strong_rgb.copy())

        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)

        if "360x_frames" in depth_name:
            sky_mask_name = depth_name.replace("360x_frames", "360x_masks")
            sky_mask = np.load(sky_mask_name)
            sky_mask = cv2.resize(sky_mask, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
            sky_mask = torch.from_numpy(sky_mask).to(torch.float32).unsqueeze(0)
            max_depth = gt_depth.max()
            if sky_mask.max() > 0:
                gt_depth[sky_mask == 1] = max_depth

        val_mask = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters)
                                & ~torch.isnan(gt_depth))

        # Normalize depth
        _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = (gt_depth - _min) / (_max - _min)
        gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)


        # Conduct output
        inputs = {}

        inputs["raw_rgb"] = self.normalize(raw_rgb)
        inputs["strong_rgb"] = self.normalize(strong_rgb)
        inputs["gt_depth"] = gt_depth_norm
        inputs["val_mask"] = val_mask

        return inputs