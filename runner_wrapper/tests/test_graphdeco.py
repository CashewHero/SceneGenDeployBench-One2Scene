from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_EXPORT_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("plyfile"))


@unittest.skipUnless(HAS_EXPORT_DEPS, "numpy and plyfile are installed in the runner image")
class GraphdecoExportTests(unittest.TestCase):
    def test_writes_full_degree_four_graphdeco_fields(self) -> None:
        import numpy as np
        from plyfile import PlyData

        from runner_wrapper.graphdeco import write_graphdeco_ply

        splats = {
            "means": np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]], dtype=np.float32),
            "scales": np.array([[0.5, 1.0, 2.0], [0.25, 0.5, 1.0]], dtype=np.float32),
            "rotations": np.array([[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "harmonics": np.arange(150, dtype=np.float32).reshape(2, 3, 25) / 100.0,
            "opacities": np.array([0.25, 0.75], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scene.ply"
            count, degree = write_graphdeco_ply(splats, output)
            vertex = PlyData.read(output)["vertex"].data

        self.assertEqual(count, 2)
        self.assertEqual(degree, 4)
        self.assertEqual(len([name for name in vertex.dtype.names if name.startswith("f_rest_")]), 72)
        self.assertAlmostEqual(float(vertex[0]["opacity"]), float(np.log(0.25 / 0.75)), places=5)
        self.assertAlmostEqual(float(vertex[0]["scale_0"]), float(np.log(0.5)), places=5)
        self.assertAlmostEqual(float(vertex[0]["rot_0"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
