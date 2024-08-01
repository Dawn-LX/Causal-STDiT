import torch
import torch.distributed as dist
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.datasets import prepare_dataloader
from opensora.datasets.skytimelapse_dataset import SkyTimelapseDataset


def skytimelapse_demo():
    
    # /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_train
    # /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test
    train_data_cfg = dict(
        type="SkyTimelapseDataset",

        root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_train",
        nframes=33,
        image_size=(128,128),
    )

    dataset:SkyTimelapseDataset = build_module(train_data_cfg, DATASETS)
    dist.init_process_group(rank=0)
    dataloader = prepare_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        process_group=dist.group.WORLD
    )
    for data in dataloader:
        for k,v in data.items():
            v = v.shape if isinstance(v,torch.Tensor) else v
            print(k,v)
        break

if __name__ == "__main__":
    skytimelapse_demo()
