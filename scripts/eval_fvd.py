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
            # the assert will be False if there are very shot videos (<self.nframes)

        
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
    _ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #

    _SKT_TIMELAPSE_ROOT = f"{_ROOT_DATA_DIR}/SkyTimelapse/sky_timelapse" 
    # /data/SkyTimelapse/sky_timelapse or /data9T/gaokaifeng/datasets/SkyTimelapse/sky_timelapse
    _VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"

    SKY_TIMELAPSE_DATASET = {
        1:dict(
            type="SkyTimelapseDatasetForEvalFVD",
            
            root=_VAL_DATA_ROOT,
            n_sample_frames = 16,
            image_size=(256,256),
            read_video = True,
            read_first_frame = False,
            class_balance_sample = True,
            num_samples_total = 2048,
        ),
        2:dict(
            type="SkyTimelapseDatasetForEvalFVD",
            
            root=_VAL_DATA_ROOT,
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
            
            root=_VAL_DATA_ROOT,
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

@torch.no_grad()
def eval_stepFVD(args):
    
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
    
    sample_config = Config.fromfile(args.sample_config)
    gen_video_dir =  sample_config.sample_save_dir
    logger.info(f"load gen data sampling config: {args.sample_config}")
    num_files = len(os.listdir(gen_video_dir))
    logger.info(f"gen_video_dir: {gen_video_dir}; num_files={num_files}")

    # ================================================================
    # 3. build dataset & dataloader
    # ================================================================
    
    SkyTimelapseDatasetForEvalFVD
    gt_data_cfg = get_gt_dataset_configs(2)
    gt_data_cfg["num_samples_total"]=num_files
    gt_dataset = build_module(gt_data_cfg, DATASETS,print_fn=logger.info)
    # build it only to get the `transforms`

    gen_dataset = GenVideoDataset(
        gen_video_dir,
        transforms=gt_dataset.transforms,
        nframes=-1, # TODO make this configable
        print_fn=logger.info
    )
    del gt_dataset
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
    ar_steps = [0,1,2,3,4,5,6]
    ar_steps_to_fids = {0:torch.as_tensor([0])}
    for ar_id in ar_steps[1:]:
        sid = (ar_id-1)*8+1
        fids = torch.as_tensor(range(sid,sid+8))
        ar_steps_to_fids.update({ar_id:fids})
    print(ar_steps_to_fids)
    
    # ar_steps_to_fids = {
    #     0:[0,1,2,3,4,5,6,7],
    #     1:[8, 9, 10, 11, 12, 13, 14, 15],
    #     2:[16, 17, 18, 19, 20, 21, 22, 23],
    #     3:[24, 25, 26, 27, 28, 29, 30, 31],
    #     4:[32, 33, 34, 35, 36, 37, 38, 39],
    #     5:[40, 41, 42, 43, 44, 45, 46, 47],
    #     6:[48, 49, 50, 51, 52, 53, 54, 55]
    # }
    ar_steps_to_fids = {k:torch.as_tensor(v) for k,v in ar_steps_to_fids.items()}
    print(ar_steps_to_fids)
    
    gen_feats = {ar_id:[] for ar_id in ar_steps[1:]}
    for batch in tqdm(gen_dataloader,desc="compute i3d feats for gen data"):
        video_names = batch["video_name"]
        gen_videos:torch.Tensor = batch["video"] # (B,C,T,H,W), values in -1 ~ 1
        gen_videos = (gen_videos + 1) / 2.0  # -1~1 --> 0~1
        assert ar_steps_to_fids[6][-1] == gen_videos.shape[2] - 1 
        for ar_id,fids in ar_steps_to_fids.items():
            if ar_id ==0:
                continue
            fids:torch.Tensor
            if len(fids) < 16:
                # incase too short for downsample in I3D network (8 x downsample)
                fids = fids.repeat_interleave(2,dim=0) 
                # [1,2,3,4,5,6,7,8] --> [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
            chunk_i = gen_videos[:,:,fids,:,:]
            feats_i:np.ndarray = _get_fvd_feats(chunk_i)
            gen_feats[ar_id].append(feats_i)
    
    chunk1 = np.concatenate(gen_feats[1],axis=0)
    logger.info(f"chunk1:{chunk1.shape}")
    fvd_to_chunk1 = dict()
    for ar_id in ar_steps[2:]: # 2,3,4,5,6
        chunk_i = np.concatenate(gen_feats[ar_id],axis=0)
        fvd_i = frechet_distance(chunk_i,chunk1)
        fvd_to_chunk1[ar_id] = fvd_i
    logger.info(f"fvd_to_chunk1={fvd_to_chunk1}")
    logger.info(f"results saved at log_path: {log_path}")

@torch.no_grad()
def eval_stepFVDtoGT(args):
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
    

    sample_config = Config.fromfile(args.sample_config)
    gen_video_dir =  sample_config.sample_save_dir
    logger.info(f"load gen data sampling config: {args.sample_config}")
    num_files = len(os.listdir(gen_video_dir))
    logger.info(f"gen_video_dir: {gen_video_dir}; num_files={num_files}")
    
    gt_data_cfg = get_gt_dataset_configs(2)
    gt_data_cfg["num_samples_total"]=num_files
    gt_data_cfg["n_sample_frames"]=16
    logger.info(f"use gt_dataset_cfg: \n {pformat(gt_data_cfg)} \n")

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

    gen_dataset = GenVideoDataset(
        gen_video_dir,
        transforms=gt_dataset.transforms,
        nframes=-1, # TODO make this configable
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
        gt_videos:torch.Tensor = batch["video"] # (B,C,T,H,W), values in -1 ~ 1
        # gt_videos = gt_videos.repeat_interleave(2,dim=2)
        gt_videos = (gt_videos + 1) / 2.0  # -1~1 --> 0~1
        feats = _get_fvd_feats(gt_videos)
        gt_feats.append(feats)


    ar_steps = [0,1,2,3]
    ar_steps_to_fids = {
        0:[0],
        1:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        2:[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        3:[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    }
    # ar_steps_to_fids = {
    #     0:[0,1,2,3,4,5,6,7],
    #     1:[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    #     2:[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    #     3:[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    # }
    ar_steps_to_fids = {k:torch.as_tensor(v) for k,v in ar_steps_to_fids.items()}
    
    gen_feats = {ar_id:[] for ar_id in ar_steps[1:]}
    for batch in tqdm(gen_dataloader,desc="compute i3d feats for gen data"):
        video_names = batch["video_name"]
        gen_videos:torch.Tensor = batch["video"] # (B,C,T,H,W), values in -1 ~ 1
        gen_videos = (gen_videos + 1) / 2.0  # -1~1 --> 0~1
        assert ar_steps_to_fids[3][-1] == gen_videos.shape[2] - 1, f"gen_videos.shape={gen_videos.shape}"
        for ar_id,fids in ar_steps_to_fids.items():
            if ar_id ==0:
                continue
            chunk_i = gen_videos[:,:,fids,:,:]
            feats_i:np.ndarray = _get_fvd_feats(chunk_i)
            gen_feats[ar_id].append(feats_i)
    
    chunk_gt = np.concatenate(gt_feats,axis=0)
    logger.info(f"chunk_gt:{chunk_gt.shape}")
    fvd_to_gt = dict()
    for ar_id in ar_steps[1:]: # 1,2,3
        chunk_i = np.concatenate(gen_feats[ar_id],axis=0)
        fvd_i = frechet_distance(chunk_i,chunk_gt)
        fvd_to_gt[ar_id] = fvd_i
    logger.info(f"fvd_to_chunk_gt={fvd_to_gt}")
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
    parser.add_argument("--step_fvd", action='store_true')
    args = parser.parse_args()

    
    I3D_WEIGHTS_DIR="/home/gkf/project/CausalSTDiT/_backup/common_metrics_on_video_quality-main/fvd"
    I3D_WEIGHTS_DIR = os.getenv("I3D_WEIGHTS_DIR",I3D_WEIGHTS_DIR)
    # e.g., 
    # i3d_weights_path = f"{I3D_WEIGHTS_DIR}/styleganv/i3d_torchscript.pt" (for styleganv)
    # i3d_weights_path = f"{I3D_WEIGHTS_DIR}/videogpt/i3d_pretrained_400.pt" (for videogpt)

    if args.step_fvd:
        # eval_stepFVD(args)
        eval_stepFVDtoGT(args)
    else:
        main(args)

