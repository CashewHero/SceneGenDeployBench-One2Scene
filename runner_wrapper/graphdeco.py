"""Validated Graphdeco PLY export for One2Scene Gaussian tensors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def write_graphdeco_ply(splats: dict[str, Any], destination: Path) -> tuple[int, int]:
    import numpy as np
    from plyfile import PlyData, PlyElement

    arrays = {
        name: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for name, value in splats.items()
    }
    means = np.asarray(arrays["means"], dtype=np.float32)
    scales = np.asarray(arrays["scales"], dtype=np.float32)
    rotations = np.asarray(arrays["rotations"], dtype=np.float32)
    harmonics = np.asarray(arrays["harmonics"], dtype=np.float32)
    alpha = np.asarray(arrays["opacities"], dtype=np.float32).reshape(-1)
    count = means.shape[0]
    if count == 0:
        raise ValueError("One2Scene produced no Gaussians")
    expected = {
        "means": (count, 3),
        "scales": (count, 3),
        "rotations": (count, 4),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"invalid {name} shape: {arrays[name].shape}; expected {shape}")
    if harmonics.ndim != 3 or harmonics.shape[:2] != (count, 3):
        raise ValueError(f"invalid harmonics shape: {harmonics.shape}")
    coefficient_count = harmonics.shape[2]
    sh_degree = math.isqrt(coefficient_count) - 1
    if sh_degree < 0 or (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(f"invalid spherical-harmonic coefficient count: {coefficient_count}")
    if alpha.shape != (count,):
        raise ValueError(f"invalid opacities shape: {alpha.shape}")
    for name, array in {**arrays, "opacities": alpha}.items():
        if not np.isfinite(array).all():
            raise ValueError(f"One2Scene produced non-finite {name}")
    if (scales <= 0).any():
        raise ValueError("One2Scene produced non-positive Gaussian scales")

    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if (rotation_norms < 1e-12).any():
        raise ValueError("One2Scene produced a zero-length Gaussian rotation")
    rotations = rotations / rotation_norms
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacity_logits = np.log(alpha / (1.0 - alpha))
    log_scales = np.log(np.maximum(scales, 1e-12))
    dc = harmonics[..., 0]
    rest = harmonics[..., 1:].reshape(count, -1)

    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names.extend(f"f_rest_{index}" for index in range(rest.shape[1]))
    names.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    vertices = np.empty(count, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = means[:, index]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0.0
    for index, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        vertices[name] = dc[:, index]
    for index in range(rest.shape[1]):
        vertices[f"f_rest_{index}"] = rest[:, index]
    vertices["opacity"] = opacity_logits
    for index, name in enumerate(("scale_0", "scale_1", "scale_2")):
        vertices[name] = log_scales[:, index]
    for index, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        vertices[name] = rotations[:, index]
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(destination)
    return count, sh_degree
