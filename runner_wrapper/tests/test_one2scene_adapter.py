from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runner_wrapper import adapter


class One2SceneAdapterTests(unittest.TestCase):
    def request(self, workspace: Path) -> dict:
        return {
            "contract_version": 1,
            "job": {
                "job_id": "job-1",
                "batch_id": "batch-1",
                "job_type": "generation",
                "primary_sample": "sample-1",
                "primary_sample_metadata": {
                    "projection": "equirectangular",
                    "fov": [360, 180],
                },
                "timeout_seconds": 3600,
                "parameters": {},
            },
            "inputs": {"data": {"sample-1": {"image": "/data/input.png"}}},
            "runtime": {"workspace_dir": str(workspace)},
        }

    def test_rejects_unknown_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown job parameters"):
            adapter.parameters({"cube_size": 512})

    def test_rejects_non_equirectangular_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be equirectangular"):
            adapter._validate_panorama_metadata(
                {"primary_sample_metadata": {"projection": "perspective"}}
            )

    def test_checkpoint_digest_is_verified_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.ckpt"
            checkpoint.write_bytes(b"test-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with (
                patch.object(adapter, "CHECKPOINT_SIZE", checkpoint.stat().st_size),
                patch.object(adapter, "CHECKPOINT_SHA256", digest),
            ):
                adapter._verify_checkpoint(checkpoint, root / "markers")
                marker = root / "markers" / f"{digest}.verified"
                self.assertTrue(marker.is_file())
                adapter._verify_checkpoint(checkpoint, root / "markers")

    def test_successful_job_reports_3dgs_and_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            checkpoint = workspace / "checkpoint.ckpt"
            checkpoint.touch()

            def export(_splats: object, destination: Path) -> tuple[int, int]:
                destination.write_bytes(b"ply")
                return 393_216, 4

            with (
                patch.object(
                    adapter,
                    "prepare_image",
                    return_value=("sample-1", Path("/data/input.png"), (1920, 960)),
                ),
                patch.object(adapter, "configure_model_cache", return_value=workspace),
                patch.object(adapter, "ensure_checkpoint", return_value=checkpoint),
                patch("runner_wrapper.one2scene_model.run_scaffold", return_value={}),
                patch("runner_wrapper.graphdeco.write_graphdeco_ply", side_effect=export),
            ):
                result = adapter.run_job(self.request(workspace))

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                result["output_files"],
                {"sample-1": {"3dgs": "3DGS-scaffold-44136fa355.ply"}},
            )
            self.assertEqual(result["output_metadata"], adapter.OUTPUT_METADATA)
            self.assertTrue((workspace / "3DGS-scaffold-44136fa355.ply").is_file())
            report_path = workspace / "metrics-scaffold-44136fa355.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["model_metrics"][0]["value"], 393_216)
            self.assertEqual(report["model_metrics"][1]["value"], 4)


if __name__ == "__main__":
    unittest.main()
