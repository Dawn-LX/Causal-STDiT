
import argparse
import hashlib
import os
import gc
from datetime import timedelta,datetime
import json
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
import torchvision

from mmengine.config import Config
from colossalai.utils import get_current_device, set_seed
from opensora.datasets import save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger
from opensora.utils.misc import (
    to_torch_dtype,
    load_jsonl
)
from opensora.utils.video_gen import autoregressive_sample,autoregressive_sample_kv_cache
from opensora.utils.debug_utils import envs
'''
        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
            dist.barrier()
'''



@torch.no_grad()
def main(cfg):

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    def is_master():
        return dist.get_rank() == 0

    # 2.2. init logger, tensorboard
    exp_dir = cfg.exp_dir
    
    os.makedirs(exp_dir,exist_ok=True)
    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory created at {exp_dir}")
    
    # make sample output dir
    exp_name = exp_dir.split('/')[-1]
    md5_tag = hashlib.md5(str(cfg._cfg_dict).encode('utf-8')).hexdigest()
    md5_tag = md5_tag + "_" + exp_name
    sample_save_dir = os.path.join(cfg.sample_save_dir,md5_tag) # maybe another disk
    os.makedirs(sample_save_dir,exist_ok=True)
    logger.info(f"sample_save_dir is:  {sample_save_dir}")

    # backup sample configs:
    _backup_path = os.path.join(exp_dir, f"sampling_cfg_{md5_tag}.json")
    cfg.update(dict(
        local_time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        utc_time_now = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S"),
        sample_save_dir = sample_save_dir
    ))
    with open(_backup_path, "w") as f:
        json.dump(cfg._cfg_dict, f, indent=4)
    logger.info(f"Backup sampling config at {_backup_path}")


    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.model.caption_channels
        text_encoder_model_max_length = 0
    
    vae = build_module(cfg.vae, MODELS)
    
    input_size = (1, cfg.sample_cfgs.width, cfg.sample_cfgs.height)
    latent_size = vae.get_latent_size(input_size)
    assert os.path.exists(cfg.ckpt_path)
    cfg.model.from_pretrained = cfg.ckpt_path
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder_output_dim,
        model_max_length=text_encoder_model_max_length
    )
    if text_encoder is not None:
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance


    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)
    model.eval()
    

    # 6.2. build validation dataset
    dataset = build_module(cfg.val_data_cfg, DATASETS)
    
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.num_workers,
        sampler=sampler
    )

    # ==========================================================================================
    # ddp sample loop
    # ==========================================================================================

    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    vae.micro_batch_size = 33
    
    scheduler = build_module(cfg.scheduler,SCHEDULERS)
    assert cfg.get("clean_prefix",True), "TODO add code for non first frame conditioned"
    assert cfg.get("clean_prefix_set_t0",True)
    if enable_kv_cache := cfg.get("enable_kv_cache",False):
        sample_func = autoregressive_sample_kv_cache
        additional_kwargs = dict(
            kv_cache_dequeue = cfg.kv_cache_dequeue,
            kv_cache_max_seqlen = cfg.kv_cache_max_seqlen,
        )
        # `kv_cache_max_seqlen` serves as `max_condion_frames` for sampling w/ kv-cache
    else:
        sample_func = autoregressive_sample
        additional_kwargs = dict(
            max_condion_frames = cfg.max_condion_frames
        )
    additional_kwargs.update(dict(
        progressive_alpha=cfg.get("progressive_alpha",-1),
        prefix_perturb_t = cfg.get("prefix_perturb_t",-1)
    ))

    
    sample_cfgs = cfg.sample_cfgs
    input_size = (sample_cfgs.auto_regre_chunk_len, sample_cfgs.height, sample_cfgs.width)
    latent_size = vae.get_latent_size(input_size)
    z_size = (vae.out_channels, *latent_size)
    auto_regre_steps = sample_cfgs.auto_regre_steps

    for batch in tqdm(dataloader,disable=not is_master()):
        
        video_names = batch["video_name"]
        prompts = batch["text"] if text_encoder is not None else [None]*len(video_names)
        if dataset.read_first_frame:
            first_frame = batch["first_frame"]  # (B, C, 1, H, W)
            first_frame = first_frame.to(device=device,dtype=dtype)
            first_frame_latents = vae.encode(first_frame) # vae accept shape (B,C,T,H,W)
        else:
            assert dataset.read_video
            video = batch["video"] # (B, C, T, H, W)
            cond_len = cfg.max_condion_frames  # e.g., T//2
            cond_video = video[:,:,:cond_len,:,:].to(device=device,dtype=dtype)
            first_frame_latents = vae.encode(cond_video) # vae accept shape (B,C,T,H,W)

        
        current_seed = sample_cfgs.seed
        if current_seed == "random":
            current_seed = int(str(datetime.now().timestamp()).split('.')[-1][:4])

        samples,time_used,num_gen_frames = sample_func(
            scheduler, 
            model, 
            text_encoder, 
            z_size = z_size, 
            prompts = prompts,
            cond_frame_latents=first_frame_latents, # (B,C,1,H,W)
            ar_steps = auto_regre_steps,
            seed = current_seed,
            verbose=False,
            **additional_kwargs
        ) # (1, C, T, H, W)
        # fps = num_gen_frames / time_used
        # print(f"num_gen_frames={num_gen_frames}, time_used={time_used:.2f}, fps={fps:.2f}")
        
        samples = vae.decode(samples.to(dtype=dtype)) # (B, C, T, H, W)
        for rank_id in range(dist.get_world_size()): # Write files sequentially
            for idx in range(samples.shape[0]):
                video_name = video_names[idx] # e.g., 07U1fSrk9oI_frames_00000046.jpg.mp4
                sample = samples[idx]
                save_path = os.path.join(sample_save_dir,video_name)
                save_sample(sample.clone(),fps=8,save_path=save_path)
                logger.info(f"rank-{rank_id} wirte video to {save_path}")

            dist.barrier()
        
        
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()



def merge_args(cfg,train_cfg,args):
    cfg.update(dict(
        model = train_cfg.model,
        vae = train_cfg.vae,
        text_encoder = train_cfg.text_encoder,
        clean_prefix = train_cfg.clean_prefix,
        clean_prefix_set_t0 = train_cfg.clean_prefix_set_t0
    ))
    assert args.ckpt_path is not None
    cfg.model.from_pretrained = args.ckpt_path

    if "prefix_perturb_t" in cfg.keys():
        # maybe change the prefix_perturb_t at test time
        pass
    else:
        cfg.update(prefix_perturb_t = train_cfg.prefix_perturb_t)

    default_cfgs = dict(
        dtype = "fp16",
        batch_size = 2,
        verbose = True,
        enable_kv_cache = True,
        kv_cache_dequeue = True,
        kv_cache_max_seqlen = 65,
    )

    for k, v in default_cfgs.items():
        if k not in cfg:
            cfg[k] = v
    
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v


    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="./configs/default.py",help="training config")
    parser.add_argument("--train_config",type=str, default="./configs/default.py",help="training config")
    parser.add_argument("--ckpt_path",type=str, default="/path/to/ckpt",help="training config")
    parser.add_argument("--exp_dir",type=str, default="/data/CausalSTDiT_working_dir/debug")
    parser.add_argument("--sample_save_dir",type=str, default="/data/sample_outputs")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    configs = Config.fromfile(args.config)
    train_configs = Config.fromfile(args.train_config)
    configs = merge_args(configs,train_configs,args)

    main(configs)
