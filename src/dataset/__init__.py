from dataclasses import fields

from torch.utils.data import Dataset

from .dataset_scannet_pose import DatasetScannetPose, DatasetScannetPoseCfgWrapper
from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg, DatasetRE10kCfgWrapper, DatasetDL3DVCfgWrapper, \
    DatasetScannetppCfgWrapper
from .types import Stage
from .view_sampler import get_view_sampler
from .dataset_s2d3d import DatasetS2d3d
from .dataset_mp3d import Datasetmp3d
from .dataset_demo import DatasetDemo

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "cube": DatasetRE10k,
    "s2d3d": DatasetS2d3d,
    "mp3d": Datasetmp3d,
    "dl3dv": DatasetRE10k,
    "demo": DatasetDemo,
    "scannetpp": DatasetRE10k,
    "scannet_pose": DatasetScannetPose,
}


DatasetCfgWrapper = DatasetRE10kCfgWrapper | DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper | DatasetScannetPoseCfgWrapper
DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets
