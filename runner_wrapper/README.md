# One2Scene scaffold runner

This wrapper adapts the feed-forward One2Scene scaffold as a SceneGenDeployBench generator.

## Contract

- Runner: `one2scene-scaffold`
- Input: one full 2:1 equirectangular `image`
- Output: one Graphdeco-compatible `3dgs` PLY with degree-4 spherical harmonics
- Native coordinates: `RDF`, with the primary panorama viewpoint at the origin
- Scene scale: `1.0` until the TartanAir calibration run supplies the release default

The wrapper uses One2Scene's active `src.dataset.utills.e2c` projection to convert the panorama to six cube faces, runs the scaffold encoder, rotates each Gaussian into the shared panorama frame, and exports Graphdeco opacity logits, log scales, normalized WXYZ quaternions, and spherical harmonics. It does not use the repository's `fused.ply`, which is an RGB point cloud rather than a 3DGS file.

`cube_size` defaults to `512`, matching the upstream demo's cubemap and camera resolution. Set it to `256` for a faster run that uses less GPU memory. These are the only supported values.

```json
{
  "parameters": {
    "cube_size": 512
  }
}
```

## Model asset

The first job downloads `one2scene_scaffold.ckpt` from the public `mutou0308/One2Scene` Hugging Face dataset into `PATH_MODEL_CACHE/one2scene`. The wrapper pins dataset revision `46367f7dc0aecfc93fb3e104ebd5994ec5731b33` and verifies SHA-256 `f833ca03e84f30e21ebbbd374b9903af22bcfa73b6523581cf77af3f984c6c05`.

Set `ONE2SCENE_CHECKPOINT` to use a preseeded checkpoint. The path must be visible inside the container. `HF_TOKEN` is optional.

## Build and test

From the repository root:

```bash
runner_wrapper/localtest.sh test
runner_wrapper/localtest.sh build
runner_wrapper/localtest.sh smoke
```

The smoke command mounts `runner_wrapper/data` by default and copies `demo_case/panorama.png` into the smoke dataset. Override the data root with `RUNNER_DATA_DIR`.

The image supports CUDA compute capabilities 7.5, 8.0, 8.6, and 8.9 with PTX. A Turing attention fallback supports the local RTX 2080 Ti. Ampere and newer GPUs use PyTorch flash attention.

## Multiple GPUs

One panorama job uses one GPU. Upstream scaffold DDP distributes samples rather than splitting one model inference, so assigning multiple GPUs to one batch-size-one job duplicates work. On a multi-GPU host, run one runner instance per GPU so DeployBench can process jobs concurrently.

The SEVA refinement stage is not part of this runner. It produces trajectory-conditioned frames rather than a reusable updated 3DGS and requires a separate runner contract.

## Catalog

Copy `runner_wrapper/config/runners/one2scene.yaml` into the deployment's active runner configuration directory.

## Licensing

The upstream repository has no root license file. Several scaffold source files identify themselves as CC BY-NC-SA 4.0 and non-commercial. Confirm that the intended deployment and image publication comply with the upstream code and checkpoint terms before publishing.
