import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import map_coordinates

# def map_coordinates_torch(cube_faces, coordinates, order=1, mode='wrap'):
#     """
#     Args:
#         cube_faces: [6, H, W] tensor
#         coordinates: list of 3 tensors [face_indices, y_coords, x_coords]
#                     每个tensor shape都是 [512, 1024]
#     """
#     face_idx, y_coords, x_coords = coordinates


#     H, W = face_idx.shape
#     device = cube_faces.device
#     dtype = cube_faces.dtype

#     # 转换mode参数
#     grid_sample_mode = 'nearest' if order == 0 else 'bilinear'
#     padding_mode = 'reflection' if mode == 'wrap' else 'border'
    
#     # 将坐标归一化到[-1, 1]
#     y_normalized = 2.0 * y_coords / (cube_faces.shape[1] - 1) - 1.0
#     x_normalized = 2.0 * x_coords / (cube_faces.shape[2] - 1) - 1.0
    
#     # 创建结果tensor
#     result = torch.zeros(H, W, device=device, dtype=dtype)
    
#     # 对每个面进行采样
#     for f in range(6):
#         mask = (face_idx == f)
#         if not mask.any():
#             continue
            
#         # 提取当前面的坐标并reshape
#         grid_y = y_normalized[mask].view(1, -1, 1)  # [1, N, 1]
#         grid_x = x_normalized[mask].view(1, -1, 1)  # [1, N, 1]
        
#         # 创建采样网格
#         grid = torch.cat([grid_x, grid_y], dim=2)  # [1, N, 2]
        
#         # 提取当前面的数据并reshape
#         face_data = cube_faces[f:f+1].view(1, 1, *cube_faces.shape[1:])  # [1, 1, H, W]
        
#         # 调整grid的shape以匹配input
#         N = grid.shape[1]
#         grid = grid.view(1, N, 1, 2)
        
#         # 采样
#         sampled = F.grid_sample(face_data, grid, 
#                               mode=grid_sample_mode,
#                               padding_mode=padding_mode,
#                               align_corners=True)
        
#         # 将结果放回对应位置
#         result[mask] = sampled.view(-1).to(dtype)
    
#     return result



# def sample_cubefaces(cube_faces):

#     device = cube_faces.device
#     dtype = cube_faces.dtype

#     tp = np.load('/home/pengfei_wang/NoPoSplat/local_pre/tp.npy')
#     coor_y = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_y.npy')
#     coor_x = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_x.npy')

#     tp = torch.from_numpy(tp).to(device=device, dtype=dtype)
#     coor_y = torch.from_numpy(coor_y).to(device=device, dtype=dtype)
#     coor_x = torch.from_numpy(coor_x).to(device=device, dtype=dtype)


#     cube_faces = cube_faces.clone()

#     cube_faces[4] = torch.rot90(cube_faces[4], k=3, dims=(0, 1))
#     cube_faces[5] = torch.rot90(cube_faces[5], k=1, dims=(0, 1))


#     pad_ud = torch.zeros((6, 2, cube_faces.shape[2]), device=device, dtype=dtype)
#     pad_ud[0, 0] = cube_faces[5, 0, :]
#     pad_ud[0, 1] = cube_faces[4, -1, :]
#     pad_ud[1, 0] = cube_faces[5, :, -1]
#     pad_ud[1, 1] = torch.flip(cube_faces[4, :, -1], [0])
#     pad_ud[2, 0] = torch.flip(cube_faces[5, -1, :], [0])
#     pad_ud[2, 1] = torch.flip(cube_faces[4, 0, :], [0])
#     pad_ud[3, 0] = torch.flip(cube_faces[5, :, 0], [0])
#     pad_ud[3, 1] = cube_faces[4, :, 0]
#     pad_ud[4, 0] = cube_faces[0, 0, :]
#     pad_ud[4, 1] = torch.flip(cube_faces[2, 0, :], [0])
#     pad_ud[5, 0] = torch.flip(cube_faces[2, -1, :], [0])
#     pad_ud[5, 1] = cube_faces[0, -1, :]
#     cube_faces = torch.cat([cube_faces, pad_ud], dim=1)

#     # Pad left right
#     pad_lr = torch.zeros((6, cube_faces.shape[1], 2), device=cube_faces.device, dtype=cube_faces.dtype)
#     pad_lr[0, :, 0] = cube_faces[1, :, 0]
#     pad_lr[0, :, 1] = cube_faces[3, :, -1]
#     pad_lr[1, :, 0] = cube_faces[2, :, 0]
#     pad_lr[1, :, 1] = cube_faces[0, :, -1]
#     pad_lr[2, :, 0] = cube_faces[3, :, 0]
#     pad_lr[2, :, 1] = cube_faces[1, :, -1]
#     pad_lr[3, :, 0] = cube_faces[0, :, 0]
#     pad_lr[3, :, 1] = cube_faces[2, :, -1]
#     pad_lr[4, 1:-1, 0] = torch.flip(cube_faces[1, 0, :], [0])
#     pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
#     pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
#     pad_lr[5, 1:-1, 1] = torch.flip(cube_faces[3, -2, :], [0])
#     cube_faces = torch.cat([cube_faces, pad_lr], dim=2)

#     return map_coordinates_torch(cube_faces, [tp, coor_y, coor_x])


def map_coordinates_torch(cube_faces, coordinates, order=1, mode='wrap'):
    """
    Args:
        cube_faces: [B, 6, H, W] tensor
        coordinates: list of 3 tensors [face_indices, y_coords, x_coords]
                    每个tensor shape都是 [512, 1024]
    """
    face_idx, y_coords, x_coords = coordinates

    H, W = face_idx.shape
    device = cube_faces.device
    dtype = cube_faces.dtype
    batch_size = cube_faces.shape[0]

    # 转换mode参数
    grid_sample_mode = 'nearest' if order == 0 else 'bilinear'
    padding_mode = 'reflection' if mode == 'wrap' else 'border'
    
    # 将坐标归一化到[-1, 1]
    y_normalized = 2.0 * y_coords / (cube_faces.shape[2] - 1) - 1.0  # [H, W]
    x_normalized = 2.0 * x_coords / (cube_faces.shape[3] - 1) - 1.0  # [H, W]
    
    # 创建结果tensor
    result = torch.zeros(batch_size, H, W, device=device, dtype=dtype)
    
    # 将cube_faces reshape为 [B*6, 1, H, W]
    cube_faces_reshaped = cube_faces.view(batch_size * 6, 1, cube_faces.shape[2], cube_faces.shape[3])
    
    # 对每个面创建mask
    all_masks = [(face_idx == f) for f in range(6)]  # 每个mask形状为 [H, W]
    
    # 对每个face的mask，生成对应的采样网格
    for f, mask in enumerate(all_masks):
        if not mask.any():
            continue
            
        # 提取当前面的坐标
        grid_y = y_normalized[mask]  # [N]
        grid_x = x_normalized[mask]  # [N]
        
        # 创建采样网格
        grid = torch.stack([grid_x, grid_y], dim=1)  # [N, 2]
        
        # 扩展网格以适应batch维度
        N = grid.shape[0]
        grid = grid.view(1, N, 1, 2).expand(batch_size, -1, -1, -1)  # [B, N, 1, 2]
        
        # 从每个batch中提取对应的face数据
        face_indices = torch.arange(batch_size, device=device) * 6 + f
        face_data = cube_faces_reshaped[face_indices]  # [B, 1, H, W]
        
        # 并行采样所有batch
        sampled = F.grid_sample(face_data, grid,
                              mode=grid_sample_mode,
                              padding_mode=padding_mode,
                              align_corners=True)  # [B, 1, N, 1]
        
        # 将结果放回对应位置
        result[:, mask] = sampled.squeeze(-1).squeeze(1)
    
    return result

def sample_cubefaces(cube_faces):
    device = cube_faces.device
    dtype = cube_faces.dtype
    batch_size = cube_faces.shape[0]  # 获取batch size

    tp = np.load('/home/pengfei_wang/NoPoSplat/local_pre/tp.npy')
    coor_y = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_y.npy')
    coor_x = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_x.npy')

    tp = torch.from_numpy(tp).to(device=device, dtype=dtype)
    coor_y = torch.from_numpy(coor_y).to(device=device, dtype=dtype)
    coor_x = torch.from_numpy(coor_x).to(device=device, dtype=dtype)

    cube_faces = cube_faces.clone()

    # 修改rot90操作以支持batch
    cube_faces[:, 4] = torch.rot90(cube_faces[:, 4], k=3, dims=(1, 2))
    cube_faces[:, 5] = torch.rot90(cube_faces[:, 5], k=1, dims=(1, 2))
    # cube_faces[:, 4] = cube_faces[:, 4].rot90(k=3, dims=(1, 2))
    # cube_faces[:, 5] = cube_faces[:, 5].rot90(k=1, dims=(1, 2))
    # for b in range(batch_size):
    #     cube_faces[b, 4] = torch.rot90(cube_faces[b, 4], k=3, dims=(0, 1))
    #     cube_faces[b, 5] = torch.rot90(cube_faces[b, 5], k=1, dims=(0, 1))

    # 修改padding操作以支持batch
    pad_ud = torch.zeros((batch_size, 6, 2, cube_faces.shape[3]), device=device, dtype=dtype)
    pad_ud[:, 0, 0] = cube_faces[:, 5, 0, :]
    pad_ud[:, 0, 1] = cube_faces[:, 4, -1, :]
    pad_ud[:, 1, 0] = cube_faces[:, 5, :, -1]
    pad_ud[:, 1, 1] = torch.flip(cube_faces[:, 4, :, -1], [1])
    pad_ud[:, 2, 0] = torch.flip(cube_faces[:, 5, -1, :], [1])
    pad_ud[:, 2, 1] = torch.flip(cube_faces[:, 4, 0, :], [1])
    pad_ud[:, 3, 0] = torch.flip(cube_faces[:, 5, :, 0], [1])
    pad_ud[:, 3, 1] = cube_faces[:, 4, :, 0]
    pad_ud[:, 4, 0] = cube_faces[:, 0, 0, :]
    pad_ud[:, 4, 1] = torch.flip(cube_faces[:, 2, 0, :], [1])
    pad_ud[:, 5, 0] = torch.flip(cube_faces[:, 2, -1, :], [1])
    pad_ud[:, 5, 1] = cube_faces[:, 0, -1, :]
    cube_faces = torch.cat([cube_faces, pad_ud], dim=2)

    # 修改左右padding以支持batch
    pad_lr = torch.zeros((batch_size, 6, cube_faces.shape[2], 2), device=cube_faces.device, dtype=cube_faces.dtype)
    pad_lr[:, 0, :, 0] = cube_faces[:, 1, :, 0]
    pad_lr[:, 0, :, 1] = cube_faces[:, 3, :, -1]
    pad_lr[:, 1, :, 0] = cube_faces[:, 2, :, 0]
    pad_lr[:, 1, :, 1] = cube_faces[:, 0, :, -1]
    pad_lr[:, 2, :, 0] = cube_faces[:, 3, :, 0]
    pad_lr[:, 2, :, 1] = cube_faces[:, 1, :, -1]
    pad_lr[:, 3, :, 0] = cube_faces[:, 0, :, 0]
    pad_lr[:, 3, :, 1] = cube_faces[:, 2, :, -1]
    pad_lr[:, 4, 1:-1, 0] = torch.flip(cube_faces[:, 1, 0, :], [1])
    pad_lr[:, 4, 1:-1, 1] = cube_faces[:, 3, 0, :]
    pad_lr[:, 5, 1:-1, 0] = cube_faces[:, 1, -2, :]
    pad_lr[:, 5, 1:-1, 1] = torch.flip(cube_faces[:, 3, -2, :], [1])
    cube_faces = torch.cat([cube_faces, pad_lr], dim=3)

    # map_coordinates_torch函数也需要相应修改以支持batch维度
    return map_coordinates_torch(cube_faces, [tp, coor_y, coor_x])


# def cube_list2h(cube_list):
#     assert len(cube_list) == 6
#     assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6

#     cube_list[1] = np.flip(cube_list[1], axis=1)
#     cube_list[2] = np.flip(cube_list[2], axis=1)
#     cube_list[4] = np.flip(np.rot90(cube_list[4], k=-1, axes=(0, 1)), axis=0)
#     cube_list[5] = np.rot90(cube_list[5], k=1, axes=(0, 1))


#     return np.concatenate(cube_list, axis=1)

def sample_cubefaces_np(cube_faces):
    tp = np.load('/home/pengfei_wang/NoPoSplat/local_pre/tp.npy')
    coor_y = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_y.npy')
    coor_x = np.load('/home/pengfei_wang/NoPoSplat/local_pre/coor_x.npy')

    cube_faces = cube_faces.cpu().numpy().copy()

    # cube_faces[1] = np.flip(cube_faces[1], 1)
    # cube_faces[2] = np.flip(cube_faces[2], 1)
    # cube_faces[4] = np.flip(cube_faces[4], 0)

    cube_faces[4] = np.rot90(cube_faces[4], k=-1, axes=(0, 1))
    cube_faces[5] = np.rot90(cube_faces[5], k=1, axes=(0, 1))


    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=1, mode='wrap')
