from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from .util import Equirec2Cube
from .e2c_utills import e2c

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



def cassini2Cube(cassini, cube_size=256):
    """
    Convert Cassini projection to Cube Map
    Args:
        cassini: Input Cassini projection image (numpy array or torch tensor)
        cube_size: Size of each face of the cube map
    Returns:
        cube_map: Output cube map with 6 faces [front, right, back, left, top, bottom]
    """
    # Input handling
    if cassini.ndim == 2:
        cassini = np.expand_dims(cassini, axis=-1)
    source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3)
    
    # Calculate grid coordinates for each face
    faces = ['front', 'right', 'back', 'left', 'top', 'bottom']
    cube_faces = []
    
    for face in faces:
        # Generate meshgrid for the face
        x_range = torch.linspace(-1, 1, cube_size)
        y_range = torch.linspace(-1, 1, cube_size)
        Y, X = torch.meshgrid(y_range, x_range)
        
        # Initialize grid coordinates
        x = X.clone()
        y = Y.clone()
        z = torch.ones_like(X)
        
        # Adjust coordinates based on face
        if face == 'front':
            z = torch.ones_like(X)
        elif face == 'right':
            temp = z.clone()
            z = -x
            x = temp
        elif face == 'back':
            z = -torch.ones_like(X)
            x = -x
        elif face == 'left':
            temp = z.clone()
            z = x
            x = -temp
        elif face == 'top':
            temp = z.clone()
            z = -y
            y = temp
        elif face == 'bottom':
            temp = z.clone()
            z = y
            y = -temp
            
        # Convert to spherical coordinates
        rho = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2(y, x)
        phi = torch.acos(z/rho)
        
        # Convert to Cassini coordinates
        theta_cassini = torch.atan2(torch.tan(phi), torch.cos(theta))
        phi_cassini = torch.asin(torch.cos(phi) * torch.sin(theta))
        
        # Create sampling grid
        grid_x = torch.stack([-phi_cassini / (0.5 * np.pi)], dim=-1)
        grid_y = torch.stack([-theta_cassini / np.pi], dim=-1)
        grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Sample the face
        face_map = F.grid_sample(source_image, grid, 
                               mode='bilinear', 
                               align_corners=True,
                               padding_mode='border')
        
        cube_faces.append(face_map)
    
    # Stack all faces
    cube_map = torch.cat(cube_faces, dim=0)
    
    # Convert back to numpy and handle dimensions
    if cassini.ndim == 3:
        cube_map = cube_map.transpose(1, 3).transpose(1, 2).data.numpy().astype(cassini.dtype)
    else:
        cube_map = cube_map.numpy()
        cube_map = cube_map.squeeze(1)
        
    return cube_map

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Deep360(data.Dataset):
    """The Deep360 Dataset"""

    def __init__(self, root_dir, list_file, height=504, width=1008, color_augmentation=True,
                 LR_filp_augmentation=True, yaw_rotation_augmentation=True, repeat=1, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Deep360 Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)

        self.w = 1024
        self.h = 512

        self.max_depth_meters = 100.0
        self.min_depth_meters = 0.01

        self.color_augmentation = color_augmentation
        self.LR_filp_augmentation = LR_filp_augmentation
        self.yaw_rotation_augmentation = yaw_rotation_augmentation

        self.is_training = is_training

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

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)
        self.e2c = e2c

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = self.root_dir + self.rgb_depth_list[idx][0]
        rgb = cv2.imread(rgb_name)
        if rgb is None:
            print(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cassini2Equirec(rgb)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # cube_my = cassini2Cube(rgb)
        depth_name = self.root_dir + self.rgb_depth_list[idx][1]
        gt_depth = np.load(depth_name)['arr_0'].astype(np.float32)
        gt_depth = cassini2Equirec(gt_depth)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb.copy()

        cube_aug_rgb, _ = self.e2c(aug_rgb)
        images_pers = []
        images_pers_resize = []
        for i in range(cube_aug_rgb.shape[0]):
            img_tensor = self.to_tensor(cube_aug_rgb[i]) 
            images_pers.append(img_tensor)

            img_resize_tensor = self.to_tensor(cv2.resize(cube_aug_rgb[i], dsize=(504, 504), interpolation=cv2.INTER_NEAREST))
            images_pers_resize.append(img_resize_tensor)
        images_pers = torch.stack(images_pers)
        images_pers_resize = torch.stack(images_pers_resize)

        aug_rgb = self.to_tensor(aug_rgb.copy())
        # aug_cube_rgb = self.to_tensor(aug_cube_rgb.copy())

        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)

        val_mask = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters) & ~torch.isnan(gt_depth))

        # Normalize depth
        _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = (gt_depth - _min) / (_max - _min)
        gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)

        # Conduct output
        inputs = {}

        inputs["rgb"] = self.normalize(aug_rgb)
        inputs["cube_rgb_o"] = self.normalize(images_pers)
        inputs["cube_rgb"] = self.normalize(images_pers_resize)
        inputs["gt_depth"] = gt_depth_norm
        inputs["val_mask"] = val_mask

        return inputs
    
if __name__ == "__main__":
    dataset = Deep360(root_dir='/hpc2hdd/home/zcao740/Documents/Dataset/Deep360', list_file='/hpc2hdd/home/zcao740/Documents/360Depth/Semi-supervision/datasets/deep360_train.txt')
    print(len(dataset))
    for i in range(2):
        print(dataset[i])