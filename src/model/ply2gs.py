import torch
import numpy as np
import struct
from pathlib import Path
from typing import Optional
from jaxtyping import Float
# 导入项目中正确的Gaussians类型
from src.model.types import Gaussians

def read_3dgs_ply_to_gaussians(file_path: str, device: str = "cuda") -> Gaussians:
    """
    从3DGS PLY文件读取数据并转换为项目期望的Gaussians格式
    
    Args:
        file_path: PLY文件路径
        device: 张量设备 ("cuda" 或 "cpu")
    
    Returns:
        Gaussians: 项目中定义的Gaussians对象
    """
    
    # 读取PLY文件
    raw_data = read_ply_file(file_path)
    
    # 提取基本数据
    N = len(raw_data['x'])  # 高斯点数量
    
    # 1. means: 位置坐标 [1, N, 3]
    means = torch.stack([
        torch.tensor(raw_data['x'], dtype=torch.float32),
        torch.tensor(raw_data['y'], dtype=torch.float32), 
        torch.tensor(raw_data['z'], dtype=torch.float32)
    ], dim=1).unsqueeze(0)  # [1, N, 3]
    
    # 2. covariances: 从缩放和旋转计算协方差矩阵 [1, N, 3, 3]
    scales = torch.stack([
        torch.tensor(raw_data['scale_0'], dtype=torch.float32),
        torch.tensor(raw_data['scale_1'], dtype=torch.float32),
        torch.tensor(raw_data['scale_2'], dtype=torch.float32)
    ], dim=1)  # [N, 3]
    
    rotations = torch.stack([
        torch.tensor(raw_data['rot_0'], dtype=torch.float32),  # w
        torch.tensor(raw_data['rot_1'], dtype=torch.float32),  # x
        torch.tensor(raw_data['rot_2'], dtype=torch.float32),  # y
        torch.tensor(raw_data['rot_3'], dtype=torch.float32)   # z
    ], dim=1)  # [N, 4]
    
    # 归一化四元数
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)
    
    # 计算协方差矩阵
    covariances = compute_covariance_matrices(scales, rotations).unsqueeze(0)  # [1, N, 3, 3]
    
    # 3. harmonics: 球谐系数 [1, N, 3, d_sh]
    # 收集所有球谐系数
    sh_dc_keys = [f'f_dc_{i}' for i in range(3)]  # DC分量 (3个)
    sh_rest_keys = []
    
    # 查找所有f_rest_键
    for key in raw_data.keys():
        if key.startswith('f_rest_'):
            sh_rest_keys.append(key)
    sh_rest_keys = sorted(sh_rest_keys)
    
    # DC分量
    sh_dc = torch.stack([
        torch.tensor(raw_data[key], dtype=torch.float32) for key in sh_dc_keys
    ], dim=1)  # [N, 3]
    
    # 其他分量
    if sh_rest_keys:
        sh_rest = torch.stack([
            torch.tensor(raw_data[key], dtype=torch.float32) for key in sh_rest_keys
        ], dim=1)  # [N, rest_count]
        
        # 重组为 [N, 3, d_sh] 格式
        rest_per_channel = len(sh_rest_keys) // 3
        sh_rest = sh_rest.view(N, 3, rest_per_channel)  # [N, 3, rest_per_channel]
        
        # 合并DC和其他分量
        harmonics = torch.cat([
            sh_dc.unsqueeze(2),  # [N, 3, 1]
            sh_rest  # [N, 3, rest_per_channel]
        ], dim=2)  # [N, 3, 1+rest_per_channel]
    else:
        harmonics = sh_dc.unsqueeze(2)  # [N, 3, 1]
    
    harmonics = harmonics.unsqueeze(0)  # [1, N, 3, d_sh]
    
    # 4. opacities: 不透明度 [1, N]
    opacities = torch.tensor(raw_data['opacity'], dtype=torch.float32).unsqueeze(0)  # [1, N]
    
    # 移动到指定设备
    means = means.to(device)
    covariances = covariances.to(device)
    harmonics = harmonics.to(device)
    opacities = opacities.to(device)
    
    # 使用项目中定义的Gaussians类创建对象
    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities
    )

def read_ply_file(file_path: str) -> dict:
    """读取PLY文件的原始数据"""
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as f:
        # 读取PLY头部
        line = f.readline().decode('utf-8').strip()
        if line != 'ply':
            raise ValueError("不是有效的PLY文件")
        
        # 解析头部信息
        vertex_count = 0
        properties = []
        in_header = True
        
        while in_header:
            line = f.readline().decode('utf-8').strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                in_header = False
        
        # 读取二进制数据
        data = {}
        dtype_map = {
            'float': 'f',
            'double': 'd', 
            'uchar': 'B',
            'int': 'i'
        }
        
        # 构建struct格式字符串
        format_str = '<'  # 小端序
        for prop_name, prop_type in properties:
            format_str += dtype_map.get(prop_type, 'f')
        
        struct_size = struct.calcsize(format_str)
        
        # 初始化数据列表
        for prop_name, _ in properties:
            data[prop_name] = []
        
        # 读取所有顶点数据
        for i in range(vertex_count):
            vertex_data = struct.unpack(format_str, f.read(struct_size))
            for j, (prop_name, _) in enumerate(properties):
                data[prop_name].append(vertex_data[j])
        
        # 转换为numpy数组
        for prop_name in data:
            data[prop_name] = np.array(data[prop_name])
    
    return data

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数 [N, 4] (w, x, y, z)
    
    Returns:
        R: 旋转矩阵 [N, 3, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # 构建旋转矩阵
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    return R

def compute_covariance_matrices(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    从缩放和旋转参数计算协方差矩阵
    
    Args:
        scales: 缩放参数 [N, 3]
        rotations: 四元数 [N, 4]
    
    Returns:
        covariances: 协方差矩阵 [N, 3, 3]
    """
    N = scales.shape[0]
    
    # 将缩放转换为对角矩阵
    S = torch.zeros(N, 3, 3, device=scales.device, dtype=scales.dtype)
    S[:, 0, 0] = torch.exp(scales[:, 0])  # 通常3DGS存储log(scale)
    S[:, 1, 1] = torch.exp(scales[:, 1])
    S[:, 2, 2] = torch.exp(scales[:, 2])
    
    # 获取旋转矩阵
    R = quaternion_to_rotation_matrix(rotations)  # [N, 3, 3]
    
    # 计算协方差矩阵: Σ = R * S * S^T * R^T
    RS = torch.bmm(R, S)  # [N, 3, 3]
    covariances = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]
    
    return covariances

# 使用示例
if __name__ == "__main__":
    ply_file_path = "/home/pengfei_wang/DreamScene360/output/104/point_cloud/iteration_10000/point_cloud.ply"
    
    try:
        # 读取并转换数据
        print("正在读取3DGS PLY文件...")
        gaussians = read_3dgs_ply_to_gaussians(ply_file_path, device="cuda")
        
        # 检查类型
        print(f"Gaussians类型: {type(gaussians)}")
        print(f"means形状: {gaussians.means.shape}")
        print(f"covariances形状: {gaussians.covariances.shape}")
        print(f"harmonics形状: {gaussians.harmonics.shape}")
        print(f"opacities形状: {gaussians.opacities.shape}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保正确导入了项目中的Gaussians类型")
    except Exception as e:
        print(f"其他错误: {e}")
        import traceback
        traceback.print_exc()