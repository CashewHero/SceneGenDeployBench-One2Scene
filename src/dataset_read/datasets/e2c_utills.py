import numpy as np
# from external.py360convert import utils
from PIL import Image
import cv2
import os
from scipy.ndimage import map_coordinates
import torch

def xyzcube(face_w, fov_deg=95):
    '''
    Return the xyz coordinates of the unit cube in [F R B L U D] format with 95-degree FOV.
    '''
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # 计算95度FOV对应的缩放因子
    # fov = 95.0
    half_fov_rad = np.deg2rad(fov_deg / 2)
    scale_factor = np.tan(half_fov_rad)  # 对应半角的tan值

    # Front face (z = 0.5)
    out[:, 0 * face_w:1 * face_w, [0, 1]] = grid * scale_factor
    out[:, 0 * face_w:1 * face_w, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1 * face_w:2 * face_w, [2, 1]] = grid * scale_factor
    out[:, 1 * face_w:2 * face_w, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2 * face_w:3 * face_w, [0, 1]] = grid * scale_factor
    out[:, 2 * face_w:3 * face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3 * face_w:4 * face_w, [2, 1]] = grid * scale_factor
    out[:, 3 * face_w:4 * face_w, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4 * face_w:5 * face_w, [0, 2]] = grid * scale_factor
    out[:, 4 * face_w:5 * face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5 * face_w:6 * face_w, [0, 2]] = grid * scale_factor
    out[:, 5 * face_w:6 * face_w, 1] = -0.5

    return out


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)


def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)


def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    # # Normalize u and v to [0, 1]
    # u = (u + np.pi) / (2 * np.pi)
    # v = (v + np.pi / 2) / np.pi
    return np.concatenate([u, v], axis=-1)


def uv2unitxyz(uv):
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return np.concatenate([x, y, z], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)


def coor2uv(coorxy, h, w):
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    v = -((coor_y + 0.5) / h - 0.5) * np.pi

    return np.concatenate([u, v], axis=-1)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]


def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    cube_faces[1] = np.flip(cube_faces[1], 1)
    cube_faces[2] = np.flip(cube_faces[2], 1)
    cube_faces[4] = np.flip(cube_faces[4], 0)

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

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6

    cube_list[1] = np.flip(cube_list[1], axis=1)
    cube_list[2] = np.flip(cube_list[2], axis=1)
    cube_list[4] = np.flip(np.rot90(cube_list[4], k=-1, axes=(0, 1)), axis=0)
    cube_list[5] = np.rot90(cube_list[5], k=1, axes=(0, 1))


    return np.concatenate(cube_list, axis=1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict([(k, cube_list[i])
                 for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D'])])


def cube_dict2h(cube_dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_h2dice(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face
    return cube_dice


def cube_dice2h(cube_dice):
    w = cube_dice.shape[0] // 3
    assert cube_dice.shape[0] == w * 3 and cube_dice.shape[1] == w * 4
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_h[:, i*w:(i+1)*w] = face
    return cube_h


def rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])

    return R



def e2c(e_img, face_w=256, mode='bilinear', cube_format='list'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = xyzcube(face_w, fov_deg=90)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    cubemap = np.stack([
        sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap = cube_h2dice(cubemap)
    else:
        raise NotImplementedError()

    return cubemap

def c2e(cubemap, h, w, mode='bilinear', cube_format='list'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # cube_faces = cubemap
    # face_w = cubemap.shape[1]

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    # fov = np.deg2rad(95)
    # scale = np.tan(fov / 2) / np.tan(np.pi / 4)  # 相对于90度FOV的缩放比例

    fov = 95.0
    half_fov_rad = np.deg2rad(fov / 2)
    scale_factor = np.tan(half_fov_rad)  # 与xyzcube()中的缩放一致


    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = (0.5 / scale_factor) * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -(0.5 / scale_factor) * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    # 修改上面的缩放因子
    c = (0.5 / scale_factor) * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    # 修改下面的缩放因子
    c = (0.5 / scale_factor) * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    # coor_x = (np.clip(coor_x, -0.5 * scale, 0.5 * scale) / (0.5 * scale) * 0.5 + 0.5) * face_w
    # coor_y = (np.clip(coor_y, -0.5 * scale, 0.5 * scale) / (0.5 * scale) * 0.5 + 0.5) * face_w

    equirec = np.stack([
        sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear', visualize=True):
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    assert len(e_img.shape) in (2, 3)
    h, w = e_img.shape[:2]

    try:
        h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    except:
        h_fov, v_fov = fov, fov
    in_rot = in_rot_deg * np.pi / 180

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    if len(e_img.shape) == 2:
        pers_img = sample_equirec(e_img, coor_xy, order=order)
    else:
        pers_img = np.stack([
            sample_equirec(e_img[..., i], coor_xy, order=order)
            for i in range(e_img.shape[2])
        ], axis=-1)



        # Save the visualization image
        # cv2.imwrite(vis_output_path, vis_img)

    return pers_img, uv



def e2c(e_img, face_w=512, mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    face_w:  int, the length of each face of the cubemap
    '''
    if e_img.shape[-1] == 4:
        e_img = e_img[:,:,:3]
    assert len(e_img.shape) in (2, 3)
    h, w = e_img.shape[:2]

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    # 生成立方体的 6 个面
    dtype = np.uint8 if e_img.shape[-1] == 3 else np.float32
    cubemap = np.zeros((6, face_w, face_w, 3 if e_img.shape[-1] == 3 else 1), dtype=dtype)
    coor_xy_list=[]
    for i in range(6):
        u_deg = [0, 90, 180, 270, -90, -90]
        v_deg = [0, 0, 0, 0, 90, -90]
        # face_order = ['front', 'right', 'back', 'left', 'up', 'down']

        pers_img, coor_xy = e2p(e_img, (95, 95), u_deg[i], v_deg[i], (face_w, face_w), mode=mode)

        coor_xy_list.append(coor_xy)
        cubemap[i] = pers_img

    return cubemap, coor_xy_list
