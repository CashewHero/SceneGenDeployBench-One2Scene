import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary."""
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(a.cross(b)), a)


def matrix_to_euler(
    rotations: Float[Tensor, "*batch 3 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: Float[Tensor, "*batch 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin


    noise_scale = 500  # 这个值可调
    translations = extrinsics[..., :3, 3]
    noise = torch.randn_like(translations) * noise_scale
    translations_noisy = translations + noise
    extrinsics_noisy = extrinsics.clone()
    extrinsics_noisy[..., :3, 3] = translations_noisy


    return extrinsics


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


# @torch.no_grad()
# def interpolate_extrinsics(
#     initial: Float[Tensor, "*#batch 4 4"],
#     final: Float[Tensor, "*#batch 4 4"],
#     t: Float[Tensor, " time_step"],
#     curve_height: float = 0.3,
#     eps: float = 1e-4,
# ) -> Float[Tensor, "*batch time_step 4 4"]:
#     """在两个相机位姿之间创建弧形探索路径"""
    
#     initial = initial.type(torch.float64)
#     final = final.type(torch.float64)
#     t = t.type(torch.float64)

#     # 获取batch维度
#     batch_dims = initial.shape[:-2]
    
#     initial_look = initial[..., :3, 2]
#     final_look = final[..., :3, 2]
#     initial_up = initial[..., :3, 1]
    
#     initial_origin = initial[..., :3, 3]
#     final_origin = final[..., :3, 3]
    
#     # 计算路径的控制点
#     path_direction = final_origin - initial_origin
#     path_length = torch.norm(path_direction, dim=-1, keepdim=True)
    
#     # 计算贝塞尔曲线的控制点
#     control_point1 = initial_origin + path_direction * 0.33
#     control_point2 = initial_origin + path_direction * 0.66
    
#     # 添加高度偏移
#     height_offset = initial_up * (curve_height * path_length)
#     control_point1 = control_point1 + height_offset
#     control_point2 = control_point2 + height_offset
    
#     # 计算贝塞尔曲线上的位置
#     def cubic_bezier(p0, p1, p2, p3, t):
#         t = t[..., None]  # 添加维度以便广播
#         return (1-t)**3 * p0[..., None, :] + \
#                3*(1-t)**2 * t * p1[..., None, :] + \
#                3*(1-t) * t**2 * p2[..., None, :] + \
#                t**3 * p3[..., None, :]
    
#     # 生成相机路径
#     position_interpolated = cubic_bezier(
#         initial_origin, control_point1, control_point2, final_origin, t
#     )
    
#     # 初始化look_directions张量，确保维度正确
#     look_directions = torch.zeros(*batch_dims, len(t), 3, device=initial.device, dtype=initial.dtype)
    
#     # 计算每个时间步的朝向
#     for i in range(len(t)):
#         # 计算切线方向
#         if i == 0:
#             tangent = position_interpolated[..., 1, :] - position_interpolated[..., 0, :]
#         elif i == len(t) - 1:
#             tangent = position_interpolated[..., -1, :] - position_interpolated[..., -2, :]
#         else:
#             tangent = position_interpolated[..., i+1, :] - position_interpolated[..., i-1, :]
        
#         tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + eps)
        
#         # 在初始和最终朝向之间插值
#         blend = t[i]
#         look_dir = (1-blend) * initial_look + blend * final_look
#         look_dir = look_dir / (torch.norm(look_dir, dim=-1, keepdim=True) + eps)
        
#         look_directions[..., i, :] = look_dir
    
#     # 初始化rotations张量
#     rotations = torch.zeros(*batch_dims, len(t), 3, 3, device=initial.device, dtype=initial.dtype)
    
#     # 构建每个时间步的旋转矩阵
#     for i in range(len(t)):
#         forward = look_directions[..., i, :]
        
#         # 计算右向量
#         right = torch.cross(forward, initial_up)
#         right = right / (torch.norm(right, dim=-1, keepdim=True) + eps)
        
#         # 重新计算上向量
#         up = torch.cross(right, forward)
#         up = up / (torch.norm(up, dim=-1, keepdim=True) + eps)
        
#         # 组装旋转矩阵
#         rotations[..., i, :, 0] = right
#         rotations[..., i, :, 1] = up
#         rotations[..., i, :, 2] = forward
    
#     # 构建完整的变换矩阵
#     result = torch.zeros(*batch_dims, len(t), 4, 4, device=initial.device, dtype=initial.dtype)
#     result[..., :3, :3] = rotations
#     result[..., :3, 3] = position_interpolated
#     result[..., 3, 3] = 1.0
    
#     return result.type(torch.float32)


# @torch.no_grad()
# def interpolate_extrinsics(
#     initial: Float[Tensor, "*#batch 4 4"],
#     final: Float[Tensor, "*#batch 4 4"],
#     t: Float[Tensor, " time_step"],
#     curve_height: float = 0.3,
#     lateral_offset: float = 0.2,  # 添加横向偏移
#     rotation_intensity: float = 1.2,  # 控制旋转强度
#     eps: float = 1e-4,
# ) -> Float[Tensor, "*batch time_step 4 4"]:
#     """创建更动态的相机运动轨迹"""
    
#     initial = initial.type(torch.float64)
#     final = final.type(torch.float64)
#     t = t.type(torch.float64)

#     batch_dims = initial.shape[:-2]
    
#     initial_look = initial[..., :3, 2]
#     final_look = final[..., :3, 2]
#     initial_up = initial[..., :3, 1]
    
#     initial_origin = initial[..., :3, 3]
#     final_origin = final[..., :3, 3]
    
#     # 计算路径基本方向
#     path_direction = final_origin - initial_origin
#     path_length = torch.norm(path_direction, dim=-1, keepdim=True)
    
#     # 计算横向偏移方向（垂直于路径方向和上向量）
#     side_direction = torch.cross(path_direction, initial_up)
#     side_direction = side_direction / (torch.norm(side_direction, dim=-1, keepdim=True) + eps)
    
#     # 使用更复杂的控制点布局
#     # 添加横向偏移和非对称高度变化
#     t_peaks = torch.tensor([0.3, 0.7], device=t.device, dtype=t.dtype)  # 峰值位置
    
#     def compute_offset(t_val):
#         # 创建更复杂的曲线
#         height = curve_height * torch.sin(torch.pi * t_val)
#         lateral = lateral_offset * torch.sin(2 * torch.pi * t_val)  # 两个周期的横向运动
#         return height, lateral

#     # 生成多个控制点
#     num_control_points = 5
#     control_points = []
#     for i in range(num_control_points):
#         t_val = i / (num_control_points - 1)
#         height, lateral = compute_offset(torch.tensor(t_val))
        
#         point = initial_origin + path_direction * t_val
#         point = point + initial_up * (height * path_length)
#         point = point + side_direction * (lateral * path_length)
#         control_points.append(point)

#     # 使用改进的样条插值
#     def improved_spline(control_points, t):
#         n = len(control_points) - 1
#         result = torch.zeros_like(control_points[0][..., None, :])
        
#         for i, point in enumerate(control_points):
#             # 使用基于余弦的平滑权重
#             weight = torch.cos(torch.pi * (t[..., None] * n - i))**2
#             mask = (t[..., None] * n >= i-1) & (t[..., None] * n <= i+1)
#             weight = torch.where(mask, weight, torch.zeros_like(weight))
#             weight = weight / (weight.sum(dim=-1, keepdim=True) + eps)
#             result = result + point[..., None, :] * weight[..., None]
            
#         return result

#     # 生成相机路径
#     position_interpolated = improved_spline(control_points, t)
    
#     # 计算朝向，使用更动态的插值
#     look_directions = torch.zeros(*batch_dims, len(t), 3, device=initial.device, dtype=initial.dtype)
    
#     for i in range(len(t)):
#         # 计算局部切线
#         if i == 0:
#             tangent = position_interpolated[..., 1, :] - position_interpolated[..., 0, :]
#         elif i == len(t) - 1:
#             tangent = position_interpolated[..., -1, :] - position_interpolated[..., -2, :]
#         else:
#             tangent = position_interpolated[..., i+1, :] - position_interpolated[..., i-1, :]
        
#         tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + eps)
        
#         # 使用非线性插值进行朝向混合
#         blend = t[i]
#         # 添加非线性变化使旋转更富有动感
#         blend_modified = 0.5 - 0.5 * torch.cos(torch.pi * blend * rotation_intensity)
#         look_dir = (1-blend_modified) * initial_look + blend_modified * final_look
#         look_dir = look_dir / (torch.norm(look_dir, dim=-1, keepdim=True) + eps)
        
#         # 混合切线方向来创造更自然的转向
#         final_look_dir = look_dir * 0.7 + tangent * 0.3
#         final_look_dir = final_look_dir / (torch.norm(final_look_dir, dim=-1, keepdim=True) + eps)
        
#         look_directions[..., i, :] = final_look_dir
    
#     # 构建旋转矩阵
#     rotations = torch.zeros(*batch_dims, len(t), 3, 3, device=initial.device, dtype=initial.dtype)
    
#     for i in range(len(t)):
#         forward = look_directions[..., i, :]
        
#         # 动态调整up向量以增加变化
#         blend = t[i]
#         current_up = initial_up + torch.sin(torch.pi * blend) * 0.1 * side_direction
#         current_up = current_up / (torch.norm(current_up, dim=-1, keepdim=True) + eps)
        
#         right = torch.cross(forward, current_up)
#         right = right / (torch.norm(right, dim=-1, keepdim=True) + eps)
        
#         up = torch.cross(right, forward)
#         up = up / (torch.norm(up, dim=-1, keepdim=True) + eps)
        
#         rotations[..., i, :, 0] = right
#         rotations[..., i, :, 1] = up
#         rotations[..., i, :, 2] = forward
    
#     result = torch.zeros(*batch_dims, len(t), 4, 4, device=initial.device, dtype=initial.dtype)
#     result[..., :3, :3] = rotations
#     result[..., :3, 3] = position_interpolated
#     result[..., 3, 3] = 1.0
    
#     return result.type(torch.float32)