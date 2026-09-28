"""One2Scene equirectangular panorama-to-3DGS runner adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor


MODEL_REPO_ID = "mutou0308/One2Scene"
MODEL_REPO_TYPE = "dataset"
MODEL_REVISION = "46367f7dc0aecfc93fb3e104ebd5994ec5731b33"
CHECKPOINT_NAME = "one2scene_scaffold.ckpt"
CHECKPOINT_SIZE = 2_003_484_732
CHECKPOINT_SHA256 = "f833ca03e84f30e21ebbbd374b9903af22bcfa73b6523581cf77af3f984c6c05"
DEFAULT_CUBE_SIZE = 512
SUPPORTED_CUBE_SIZES = (256, 512)
OUTPUT_METADATA = {
    # One2Scene's primary cube face uses camera axes X right, Y down, Z forward.
    # The scale remains provisional until the TartanAir calibration run is complete.
    "scene_scale": 1.0,
    "scene_coordinate_system": "RDF",
    "scene_units": "relative",
    "scene_origin": "primary_viewpoint",
}


def utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def parameters(raw: object) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("job.parameters must be an object")
    unknown = set(raw) - {"cube_size"}
    if unknown:
        raise ValueError(f"unknown job parameters: {', '.join(sorted(map(str, unknown)))}")
    cube_size = raw.get("cube_size", DEFAULT_CUBE_SIZE)
    if type(cube_size) is not int or cube_size not in SUPPORTED_CUBE_SIZES:
        choices = ", ".join(map(str, SUPPORTED_CUBE_SIZES))
        raise ValueError(f"job.parameters.cube_size must be one of: {choices}")
    return {"cube_size": cube_size}


def variant_key(params: dict[str, Any]) -> str:
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"scaffold-{hashlib.sha256(encoded).hexdigest()[:10]}"


def _normalized_inputs(raw_inputs: object) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_role, raw_samples in raw_inputs.items():
        role = str(raw_role).strip()
        if not role or not isinstance(raw_samples, dict):
            raise ValueError("each input role must contain a sample mapping")
        samples: dict[str, dict[str, Any]] = {}
        for raw_sample, raw_data in raw_samples.items():
            sample = str(raw_sample).strip()
            if not sample or not isinstance(raw_data, dict):
                raise ValueError(f"inputs.{role} must map sample ids to data mappings")
            samples[sample] = {
                str(data_type).strip(): value.strip() if isinstance(value, str) else value
                for data_type, value in raw_data.items()
                if str(data_type).strip()
            }
        normalized[role] = samples
    return normalized


def _validate_panorama_metadata(job: dict[str, Any]) -> None:
    metadata = job.get("primary_sample_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("job.primary_sample_metadata must be an object")
    projection = str(metadata.get("projection") or "equirectangular").strip().lower()
    if projection != "equirectangular":
        raise ValueError(
            "primary_sample_metadata.projection must be equirectangular; "
            f"received {projection!r}"
        )
    fov = metadata.get("fov")
    if fov is not None:
        if (
            not isinstance(fov, (list, tuple))
            or len(fov) != 2
            or any(type(value) not in (int, float) for value in fov)
            or not math.isclose(float(fov[0]), 360.0, abs_tol=1e-3)
            or not math.isclose(float(fov[1]), 180.0, abs_tol=1e-3)
        ):
            raise ValueError("One2Scene requires a full 360 by 180 degree panorama")


def prepare_image(request: dict[str, Any], destination: Path) -> tuple[str, Path, tuple[int, int]]:
    from PIL import Image, UnidentifiedImageError

    job = request.get("job")
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    if job.get("job_type") not in ("generation", "generator"):
        raise ValueError("One2Scene scaffold accepts generation jobs only")
    primary = str(job.get("primary_sample") or "").strip()
    if not primary:
        raise ValueError("job.primary_sample is required")

    inputs = _normalized_inputs(request.get("inputs"))
    samples = inputs.get("data", {})
    if set(samples) != {primary}:
        raise ValueError("One2Scene requires exactly one primary sample in inputs.data")
    if inputs.get("candidate") or inputs.get("references"):
        raise ValueError("One2Scene scaffold does not consume candidate or reference inputs")
    image_value = samples[primary].get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"inputs.data.{primary}.image must be a file path")
    source = Path(image_value)
    if not source.is_file():
        raise FileNotFoundError(f"input image not found: {source}")

    _validate_panorama_metadata(job)
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if width != 2 * height:
                raise ValueError(
                    "One2Scene requires a 2:1 equirectangular image; "
                    f"received {width}x{height}"
                )
            if width < 128 or height < 64:
                raise ValueError("input panorama must be at least 128 by 64 pixels")
            image.convert("RGB").save(destination, format="PNG")
    except UnidentifiedImageError as exc:
        raise ValueError(f"input image is not readable: {source}") from exc
    return primary, source, (width, height)


def configure_model_cache() -> Path:
    cache_root = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / "one2scene"
    cache_root.mkdir(parents=True, exist_ok=True)
    for name, relative in {
        "HF_HOME": "huggingface",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    return cache_root


def _verify_checkpoint(path: Path, marker_dir: Path) -> None:
    size = path.stat().st_size
    if size != CHECKPOINT_SIZE:
        raise ValueError(f"checkpoint size mismatch for {path}: expected {CHECKPOINT_SIZE}, got {size}")
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{CHECKPOINT_SHA256}.verified"
    marker_value = f"{path.resolve()}\n{CHECKPOINT_SIZE}\n"
    if marker.is_file() and marker.read_text(encoding="utf-8") == marker_value:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch for {path}")
    temporary = marker.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(marker_value, encoding="utf-8")
    os.replace(temporary, marker)


def ensure_checkpoint(cache_root: Path) -> Path:
    override = os.getenv("ONE2SCENE_CHECKPOINT", "").strip()
    if override:
        checkpoint = Path(override)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ONE2SCENE_CHECKPOINT not found: {checkpoint}")
    else:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                repo_type=MODEL_REPO_TYPE,
                filename=CHECKPOINT_NAME,
                revision=MODEL_REVISION,
                cache_dir=cache_root / "huggingface",
                token=os.getenv("HF_TOKEN") or None,
            )
        )
    _verify_checkpoint(checkpoint, cache_root / "verified")
    return checkpoint


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    runtime = request.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("workspace_dir"):
        raise ValueError("runtime.workspace_dir is required")
    workspace = Path(runtime["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    fallback = hashlib.sha256(str(request.get("job", {}).get("job_id", "job")).encode()).hexdigest()[:10]
    try:
        params = parameters(request.get("job", {}).get("parameters"))
        variant = variant_key(params)
    except (AttributeError, TypeError, ValueError):
        variant = f"invalid-{fallback}"

    log_path = workspace / f"runner-{variant}.log"
    report_path = workspace / f"metrics-{variant}.json"
    stage = "validation"
    monitor: ResourceMonitor | None = None
    metrics: list[dict[str, Any]] = []
    report: dict[str, Any] = {"inputs": request.get("inputs", {})}

    with tee_job_output(log_path):
        try:
            job = request["job"]
            params = parameters(job.get("parameters"))
            report["parameters"] = params
            prepared = workspace / "input-panorama.png"
            primary, source, resolution = prepare_image(request, prepared)
            monitor = ResourceMonitor(sample_data={"data.image": str(source)}, output_dir=workspace)
            monitor.start()

            stage = "model_assets"
            cache_root = configure_model_cache()
            checkpoint = ensure_checkpoint(cache_root)
            print(f"One2Scene checkpoint: {checkpoint}", flush=True)

            stage = "model_inference"
            from runner_wrapper.one2scene_model import run_scaffold

            splats = run_scaffold(prepared, checkpoint, cube_size=params["cube_size"])

            stage = "export"
            from runner_wrapper.graphdeco import write_graphdeco_ply

            output_name = f"3DGS-{variant}.ply"
            gaussian_count, sh_degree = write_graphdeco_ply(splats, workspace / output_name)
            output_files = {primary: {"3dgs": output_name}}
            output_metadata = dict(OUTPUT_METADATA)
            model_metrics = [
                {
                    "namespace": "model",
                    "name": "gaussian_count",
                    "type": "integer",
                    "value": gaussian_count,
                    "unit": "gaussians",
                    "source": "model",
                },
                {
                    "namespace": "model",
                    "name": "spherical_harmonic_degree",
                    "type": "integer",
                    "value": sh_degree,
                    "unit": "degree",
                    "source": "model",
                },
                {
                    "namespace": "model",
                    "name": "cube_face_resolution",
                    "type": "integer",
                    "value": params["cube_size"],
                    "unit": "pixels",
                    "source": "runner",
                },
                {
                    "namespace": "model",
                    "name": "checkpoint_revision",
                    "type": "string",
                    "value": MODEL_REVISION,
                    "source": "model",
                },
            ]
            report.update(
                input_resolution=list(resolution),
                output_files=output_files,
                output_metadata=output_metadata,
                model_metrics=model_metrics,
            )
            result: dict[str, Any] = {
                "status": "completed",
                "output_files": output_files,
                "output_metadata": output_metadata,
                "failure": None,
            }
        except Exception as exc:
            traceback.print_exc()
            retryable = stage == "model_assets" and isinstance(exc, (ConnectionError, OSError, TimeoutError))
            failure = {
                "code": "ONE2SCENE_FAILED",
                "message": f"{stage}: {exc}",
                "retryable": retryable,
                "stage": stage,
            }
            report["failure"] = failure
            result = {"status": "failed", "failure": failure}
        finally:
            if monitor is not None:
                metrics = monitor.stop()

        model_metrics = report.get("model_metrics", [])
        result_metrics = [*metrics, *model_metrics]
        report["resource_metrics"] = metrics
        _write_report(report_path, report)

    result.update(
        started_at=utc(started),
        completed_at=utc(time.time()),
        metrics=result_metrics,
        artifacts=[
            {"artifact_type": "job_log", "path": log_path.name},
            {"artifact_type": "metric_summary", "path": report_path.name},
        ],
    )
    return result
