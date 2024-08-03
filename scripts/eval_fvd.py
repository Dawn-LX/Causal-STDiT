import os
import argparse
import hashlib
import json
import random
import numpy as np
from tqdm import tqdm
from pprint import pformat
import torch
import torchvision
from torch.utils.data import DataLoader
from mmengine.config import Config
from opensora.registry import DATASETS, build_module
from opensora.datasets.skytimelapse_dataset import SkyTimelapseDatasetForEvalFVD
from opensora.utils.ckpt_utils import create_logger

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

# reference: https://github.com/JunyaoHu/common_metrics_on_video_quality
def calculate_fvd(videos1, videos2, device, method='styleganv'):
    '''
    videos: [batch_size, timestamps, channel, h, w] value range in [0,1]
    
    '''
    assert videos1.shape == videos2.shape

    if method == 'styleganv':
        from evaluation.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
        i3d_weights_path = "/home/gkf/project/CausalSTDiT/_backup/common_metrics_on_video_quality-main/fvd/styleganv/i3d_torchscript.pt"
    elif method == 'videogpt':
        from evaluation.fvd.videogpt.fvd import load_i3d_pretrained
        from evaluation.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from evaluation.fvd.videogpt.fvd import frechet_distance
        i3d_weights_path = "/home/gkf/project/CausalSTDiT/_backup/common_metrics_on_video_quality-main/fvd/videogpt/i3d_pretrained_400.pt"
    

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(weights_path=i3d_weights_path,device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
       
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features (`get_fvd_feats` accepts values in [0,1])
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device) # (bsz, 400); values in [0,1]
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
      
        # calculate FVD when timestamps[:clip]
        
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2) #

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result

class GenVideoDataset:
    def __init__(
        self,
        video_dir,
        transforms=None,
        nframes=-1,
        nsamples=-1,
        print_fn = print
    ):
        self.video_dir = video_dir
        self.transforms = transforms  if transforms is not None else lambda x:x
        self.nframes = nframes
        self.nsamples = nsamples
        self.print_fn = print_fn

        filenames = os.listdir(video_dir)
        random.shuffle(filenames) # make sure it is random, although os.listdir is not sorted
        if nsamples > 0:
            filenames = filenames[:nsamples]
            if len(filenames) < nsamples:
                print_fn(f"NOTE: {nsamples} samples is reuqired, but gen data only has {len(filenames)} samples in dir:{video_dir}")
            else:
                print_fn(f"{nsamples} samples random selected form gen data dir:{video_dir}")
        video_paths = [os.path.join(video_dir,fn) for fn in filenames]
        self.video_paths = video_paths
    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self,idx):
        path = self.video_paths[idx]
        video,audio,info = torchvision.io.read_video(path,pts_unit='sec',output_format="THWC") # uint8 in [0,255]
        video_fps = info["video_fps"]

        video = video.permute(0,3,1,2) # THWC -> TCHW
        video = self.transforms(video) # TCHW,  values in -1 ~ 1
        
        if self.nframes > 0:
            video = video[:self.nframes]
            assert video.shape[0] == self.nframes # otherwise we cannot stack them in `collate_fn`

        
        sample = {
            "video_name":path.split('/')[-1],
            "video":video.permute(1,0,2,3), # TCHW -> CTHW,  values in -1 ~ 1
        }

        return sample


def gen_data_demo():
    video_dir = "/data/sample_outputs/14831e3e05cfd0b1d0a97c2ff2a6b3f5_debug_inference"
    gen_dataset = GenVideoDataset(video_dir,nframes=16,nsamples=9999)
    gen_dataloader = DataLoader(
        gen_dataset,
        batch_size=4,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )
    for batch in gen_dataloader:
        for k,v in batch.items():
            v = v.shape if isinstance(v,torch.Tensor) else v
            print(k,v)
        break
    print(len(gen_dataset))

def get_gt_dataset_configs(idx):
    SKY_TIMELAPSE_DATASET = {
        1:dict(
            type="SkyTimelapseDatasetForEvalFVD",
            
            root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test",
            n_sample_frames = 16,
            image_size=(256,256),
            read_video = True,
            read_first_frame = False,
            class_balance_sample = True,
            num_samples_total = 2048,
        ),
        2:dict(
            type="SkyTimelapseDatasetForEvalFVD",
            
            root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test",
            n_sample_frames = 16,
            image_size=(256,256),
            read_video = True,
            read_first_frame = False,
            class_balance_sample = True,
            long_vid_as_class = False,
            num_samples_total = 2048,
        ),
        3:dict(
            type="SkyTimelapseDatasetForEvalFVD",
            
            root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test",
            n_sample_frames = 32,
            image_size=(256,256),
            read_video = True,
            read_first_frame = False,
            class_balance_sample = True,
            num_samples_total = 2048,
        ),
    }
    # TODO add other datasets
    return SKY_TIMELAPSE_DATASET[idx]

@torch.no_grad()
def main(args):
    # ================================================================
    # 1. prepare i3d & fvd related funcs
    # ================================================================
    if args.method == 'styleganv':
        from evaluation.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
        i3d_weights_path = f"{I3D_WEIGHTS_DIR}/styleganv/i3d_torchscript.pt"
    elif args.method == 'videogpt':
        from evaluation.fvd.videogpt.fvd import load_i3d_pretrained
        from evaluation.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from evaluation.fvd.videogpt.fvd import frechet_distance
        i3d_weights_path = f"{I3D_WEIGHTS_DIR}/videogpt/i3d_pretrained_400.pt"


    device=torch.device("cuda")
    i3d = load_i3d_pretrained(weights_path=i3d_weights_path,device=device)

    def _get_fvd_feats(vid:torch.Tensor) -> np.ndarray :
        # vid: (B,C,T,H,W)
        bsz = vid.shape[0] # we use the bsz outside (i.e., the bsz of dataloader)
        feats = get_fvd_feats(vid, i3d=i3d, device=device,bs=bsz) # (bsz,400)
        if args.method == 'styleganv':
            pass
        elif args.method == 'videogpt':
            feats = feats.cpu().numpy()
        return feats
        

    # ================================================================
    # 2. prepare logger & configs
    # ================================================================

    os.makedirs(exp_dir:=args.exp_dir,exist_ok=True)
    logger,log_path = create_logger(exp_dir,return_log_path=True)
    gt_data_cfg = get_gt_dataset_configs(2)
    logger.info(f"use gt_dataset_cfg: \n {pformat(gt_data_cfg)} \n")

    sample_config = Config.fromfile(args.sample_config)
    gen_video_dir =  sample_config.sample_save_dir
    logger.info(f"load gen data sampling config: {args.sample_config}")
    logger.info(f"gen_video_dir: {gen_video_dir}")
    
    
    md5_tag = hashlib.md5(str(gt_data_cfg).encode('utf-8')).hexdigest()
    gt_feats_cache_path = md5_tag + "_gt_feats_cache.npy"
    gt_feats_cache_path = os.path.join(args.exp_dir,gt_feats_cache_path) # maybe another disk
    if os.path.exists(gt_feats_cache_path):
        pass
        # TODO read cache, skip gt_feats computation
        # But each run will random sample gt clips, so cached gt_feats is useless ?

    # ================================================================
    # 3. build dataset & dataloader
    # ================================================================
    
    SkyTimelapseDatasetForEvalFVD
    gt_dataset = build_module(gt_data_cfg, DATASETS,print_fn=logger.info)
    gt_dataloader = DataLoader(
        gt_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ''' # use first frame expand to nframes as fake gen video
    gt_data_cfg.update(dict(
        read_video = False,
        read_first_frame = True, 
    ))
    gen_dataset = build_module(gt_data_cfg, DATASETS,print_fn=logger.info)
    for data in gen_dataloader:
        gen_videos = batch["first_frame"] # (B,C,1,H,W), values in -1 ~ 1
        gen_videos = (gen_videos + 1) / 2.0  # -1~1 --> 0~1
        nframes = gt_data_cfg["n_sample_frames"]
        gen_videos = gen_videos.repeat_interleave(nframes,dim=2)
    '''

    gen_dataset = GenVideoDataset(
        gen_video_dir,
        transforms=gt_dataset.transforms,
        nframes=gt_dataset.n_sample_frames,
        nsamples = gt_dataset.num_samples_total,
        print_fn=logger.info
    )
    gen_dataloader = DataLoader(
        gen_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # ================================================================
    # 4. compute i3d feats
    # ================================================================

    gt_feats = []
    for batch in tqdm(gt_dataloader,desc="compute i3d feats for gt data"):
        video_names = batch["video_name"]
        gt_videos = batch["video"] # (B,C,T,H,W), values in -1 ~ 1
        gt_videos = (gt_videos + 1) / 2.0  # -1~1 --> 0~1
        feats = _get_fvd_feats(gt_videos)
        gt_feats.append(feats)
    gt_feats = np.concatenate(gt_feats,axis=0)

    gen_feats = []
    for batch in tqdm(gen_dataloader,desc="compute i3d feats for gen data"):
        video_names = batch["video_name"]
        gen_videos:torch.Tensor = batch["video"] # (B,C,T,H,W), values in -1 ~ 1
        gen_videos = (gen_videos + 1) / 2.0  # -1~1 --> 0~1
        feats = _get_fvd_feats(gen_videos)
        gen_feats.append(feats)
    gen_feats = np.concatenate(gen_feats,axis=0)

    logger.info(f"gt_feats:{gt_feats.shape}; gen_feats:{gen_feats.shape}")
    fvd = frechet_distance(gen_feats,gt_feats)
    logger.info(f"fvd={fvd}")
    logger.info(f"results saved at log_path: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_data_config",type=str, default="./configs/default.py",help="training config")
    parser.add_argument("--sample_config",type=str, default="./configs/default.py",help="the backuped sampling config")
    parser.add_argument("--exp_dir",type=str, default="_backup/exp_dir",help="exp_dir ")

    
    parser.add_argument("--method",type=str, default="styleganv",choices=["styleganv","videogpt"])
    parser.add_argument("--batch_size",type=int, default=2)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    
    I3D_WEIGHTS_DIR="/home/gkf/project/CausalSTDiT/_backup/common_metrics_on_video_quality-main/fvd"
    I3D_WEIGHTS_DIR = os.getenv("I3D_WEIGHTS_DIR",I3D_WEIGHTS_DIR)
    # e.g., 
    # i3d_weights_path = f"{I3D_WEIGHTS_DIR}/styleganv/i3d_torchscript.pt" (for styleganv)
    # i3d_weights_path = f"{I3D_WEIGHTS_DIR}/videogpt/i3d_pretrained_400.pt" (for videogpt)

    main(args)
    # gen_data_demo()

    '''fvd results:

    first frame as fake gen video: 
        32 frames 256x256:  366.5194737086782
        16 frames 256x256:  186.49447370742; 191.71750748020247; 174.61382794809907; 179.20583738264776
    
    /data/sample_outputs/14831e3e05cfd0b1d0a97c2ff2a6b3f5_debug_inference
    16x256x256 fvd=180.3370887303169
    '''
