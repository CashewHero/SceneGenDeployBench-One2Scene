"""Lean One2Scene scaffold inference without the Lightning demo pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any


DEFAULT_CUBE_SIZE = 512
CUBE_FACE_COUNT = 6


def _install_upstream_package_shells() -> None:
    """Load inference modules without running the training package initializers."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    packages = {
        "src.dataset": source_root / "dataset",
        "src.model.encoder": source_root / "model" / "encoder",
    }
    for name, path in packages.items():
        if name in sys.modules:
            continue
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package


def _rotation_matrix(radians: float, axis: Any) -> Any:
    import numpy as np

    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    return cosine * np.eye(3, dtype=np.float32) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross


def _camera(fov_degrees: float, theta_degrees: float, phi_degrees: float, size: int) -> tuple[Any, Any]:
    import numpy as np

    focal = 0.5 * size / np.tan(0.5 * np.radians(fov_degrees))
    intrinsic = np.array(
        [
            [focal / size, 0.0, (size - 1) / (2.0 * size)],
            [0.0, focal / size, (size - 1) / (2.0 * size)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    yaw = _rotation_matrix(np.radians(theta_degrees), [0.0, 1.0, 0.0])
    pitch_axis = yaw @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pitch = _rotation_matrix(np.radians(phi_degrees), pitch_axis)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = pitch @ yaw
    return intrinsic, camera_to_world


def panorama_to_cube_faces(image_path: Path, size: int = DEFAULT_CUBE_SIZE) -> Any:
    _install_upstream_package_shells()
    import numpy as np
    from PIL import Image
    from src.dataset.utills import e2c

    with Image.open(image_path) as source:
        panorama = np.asarray(source.convert("RGB")).copy()
    faces, _ = e2c(panorama, face_w=size, mode="bilinear")
    return np.asarray(faces, dtype=np.float32) / 255.0


def _model_config() -> Any:
    _install_upstream_package_shells()
    from src.model.encoder.backbone.backbone_fast3r import BackboneFast3rCfg
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from src.model.encoder.encoder_fastsplat import EncoderFastSplatCfg, OpacityMappingCfg
    from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

    return EncoderFastSplatCfg(
        name="fastsplat",
        d_feature=128,
        num_monocular_samples=32,
        backbone=BackboneFast3rCfg(
            name="fast3r",
            model="ViTLarge_BaseDecoder",
            patch_embed_cls="PatchEmbedDust3R",
            asymmetry_decoder=True,
            intrinsics_embed_loc="encoder",
            intrinsics_embed_degree=4,
            intrinsics_embed_type="token",
        ),
        visualizer=EncoderVisualizerEpipolarCfg(num_samples=8, min_resolution=256, export_ply=False),
        gaussian_adapter=GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4),
        apply_bounds_shim=True,
        opacity_mapping=OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1),
        gaussians_per_pixel=1,
        num_surfaces=1,
        gs_params_head_type="dpt_gs",
        input_mean=(0.5, 0.5, 0.5),
        input_std=(0.5, 0.5, 0.5),
        pretrained_weights="",
        pose_free=False,
    )


def _load_encoder(checkpoint_path: Path, device: Any) -> Any:
    import torch

    _install_upstream_package_shells()
    from src.model.encoder.encoder_fastsplat import EncoderFastSplat

    encoder = EncoderFastSplat(_model_config())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("One2Scene checkpoint does not contain a state_dict")
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("One2Scene checkpoint contains no encoder weights")
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval().requires_grad_(False)
    return encoder.to(device)


def _use_turing_attention_fallback(encoder: Any, capability: tuple[int, int]) -> None:
    if capability[0] >= 8:
        return
    for module in encoder.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "pytorch_naive"


def _context(image_path: Path, device: Any, cube_size: int) -> dict[str, Any]:
    import numpy as np
    import torch

    faces = panorama_to_cube_faces(image_path, size=cube_size)
    images = torch.from_numpy(faces).permute(0, 3, 1, 2).contiguous()
    images = (images - 0.5) / 0.5
    angles = [(0, 0), (90, 0), (180, 0), (270, 0), (-90, 90), (-90, -90)]
    cameras = [_camera(95.0, theta, phi, cube_size) for theta, phi in angles]
    intrinsics = torch.from_numpy(np.stack([camera[0] for camera in cameras]))
    extrinsics = torch.from_numpy(np.stack([camera[1] for camera in cameras]))
    return {
        "image": images.unsqueeze(0).to(device=device, dtype=torch.float32),
        "intrinsics": intrinsics.unsqueeze(0).to(device=device, dtype=torch.float32),
        "extrinsics": extrinsics.unsqueeze(0).to(device=device, dtype=torch.float32),
    }


def _quaternion_multiply(left: Any, right: Any) -> Any:
    import torch

    lw, lx, ly, lz = torch.unbind(left, dim=-1)
    rw, rx, ry, rz = torch.unbind(right, dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _world_rotations(local_xyzw: Any, extrinsics: Any) -> Any:
    import torch
    from scipy.spatial.transform import Rotation

    face_xyzw = Rotation.from_matrix(extrinsics[0, :, :3, :3].detach().cpu().numpy()).as_quat()
    face_wxyz = torch.as_tensor(face_xyzw[:, [3, 0, 1, 2]], device=local_xyzw.device, dtype=local_xyzw.dtype)
    gaussians_per_face = local_xyzw.shape[0] // CUBE_FACE_COUNT
    if gaussians_per_face * CUBE_FACE_COUNT != local_xyzw.shape[0]:
        raise ValueError("Gaussian count is not divisible by the six cube faces")
    face_wxyz = face_wxyz.repeat_interleave(gaussians_per_face, dim=0)
    local_wxyz = local_xyzw[:, [3, 0, 1, 2]]
    world = _quaternion_multiply(face_wxyz, local_wxyz)
    return world / world.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _world_harmonics(local_harmonics: Any, extrinsics: Any) -> Any:
    import torch
    from src.misc.sh_rotation import rotate_sh

    gaussians_per_face = local_harmonics.shape[0] // CUBE_FACE_COUNT
    if gaussians_per_face * CUBE_FACE_COUNT != local_harmonics.shape[0]:
        raise ValueError("Gaussian count is not divisible by the six cube faces")
    rotated = []
    for face in range(CUBE_FACE_COUNT):
        start = face * gaussians_per_face
        end = start + gaussians_per_face
        rotated.append(
            rotate_sh(
                local_harmonics[start:end],
                extrinsics[0, face, :3, :3],
            )
        )
    return torch.cat(rotated, dim=0)


def run_scaffold(
    image_path: Path,
    checkpoint_path: Path,
    cube_size: int = DEFAULT_CUBE_SIZE,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("One2Scene scaffold requires an NVIDIA CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError("One2Scene scaffold requires CUDA compute capability 7.5 or newer")
    device = torch.device("cuda:0")
    context = _context(image_path, device, cube_size)
    encoder = _load_encoder(checkpoint_path, device)
    _use_turing_attention_fallback(encoder, capability)
    visualization: dict[str, Any] = {}
    with torch.inference_mode():
        gaussians, _ = encoder(context, 0, visualization_dump=visualization)
    rotations = _world_rotations(visualization["rotations"][0], context["extrinsics"])
    harmonics = _world_harmonics(gaussians.harmonics[0], context["extrinsics"])
    result = {
        "means": gaussians.means[0].float().cpu(),
        "scales": visualization["scales"][0].float().cpu(),
        "rotations": rotations.float().cpu(),
        "harmonics": harmonics.float().cpu(),
        "opacities": gaussians.opacities[0].float().cpu(),
    }
    del encoder, gaussians, visualization, context
    torch.cuda.empty_cache()
    return result
