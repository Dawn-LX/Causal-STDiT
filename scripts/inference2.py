from copy import deepcopy
import argparse
import random
import os
import gc
from datetime import timedelta,datetime
from pprint import pformat
from easydict import EasyDict
from tqdm import tqdm
import math

import torch
import torch.distributed as dist
import torchvision

from mmengine.config import Config

from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed


from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader,save_sample
from opensora.datasets import video_transforms

from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import create_tensorboard_writer,save_training_config
from opensora.utils.misc import (
    all_reduce_mean, 
    format_numel_str, 
    get_model_numel, 
    requires_grad, 
    to_torch_dtype,
    load_jsonl
)
from opensora.utils.train_utils import update_ema, PrefixLenSampler
from opensora.utils.debug_utils import envs

@torch.no_grad()
def main(cfg):
    # ======================================================
    # 1. args & cfg
    # ======================================================
    if cfg.fix_ar_size:
        prefix_len_sampler = PrefixLenSampler(cfg.ar_size,cfg.prefix_min_len,cfg.prefix_sampling_strategy)

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
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard
    exp_dir = cfg.outputs
    assert coordinator.is_master(), "TODO: add code to support sample_ddp"
    os.makedirs(exp_dir,exist_ok=True)
    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory created at {exp_dir}")
    logger.info(f"Training configuration:\n {pformat(cfg._cfg_dict)}")
    _backup_path = save_training_config(cfg._cfg_dict,exp_dir)
    logger.info(f"Backup training config at {_backup_path}")
    
    writer = create_tensorboard_writer(exp_dir)


    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS)
    n_sample_frames = cfg.train_data_cfg.n_sample_frames
    image_size = cfg.train_data_cfg.aspect_ratio_buckets[0]
    input_size = (n_sample_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    assert os.path.exists(cfg.ckpt_path)
    cfg.model.from_pretrained = cfg.ckpt_path
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance


    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)
    model.eval()
    

    # 6.2. build validation examples & validate before train
    val_cfgs = cfg.validation_configs
    val_cfgs.update(dict(
        clean_prefix = cfg.clean_prefix,
        clean_prefix_set_t0 = cfg.clean_prefix_set_t0,
        dtype = cfg.dtype,
    ))
    val_examples = build_validate_examples(val_cfgs.examples,val_cfgs.sample_cfgs,print_fn=logger.info)
    
    # ==========================================================================================
    # validation_visualize(model,vae,text_encoder,val_examples,val_cfgs,exp_dir,writer)
    # ==========================================================================================

    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    
    global_step = int(cfg.ckpt_path.split("global_step")[-1])
    save_dir = os.path.join(exp_dir,"val_samples_from_ckpt",f"{global_step}")
    os.makedirs(save_dir,exist_ok=True)

    val_scheduler = build_module(val_cfgs.scheduler,SCHEDULERS)
    enable_kv_cache = val_cfgs.pop("enable_kv_cache",True)
    kv_cache_dequeue = val_cfgs.pop("kv_cache_dequeue",True)
    sample_func = val_scheduler.sample_with_kv_cache if enable_kv_cache else val_scheduler.sample
    kv_cache_max_seqlen = val_cfgs.kv_cache_max_seqlen
    for idx,example in enumerate(val_examples):
        current_seed = example.seed
        if current_seed == "random":
            current_seed = int(str(datetime.now().timestamp()).split('.')[-1][:4])
        set_seed(current_seed) # TODO 要用generator seet seed 才能每个 example 由自己的seed 唯一确定，否则只是设置了起始seed，与example list的顺序有关

        if (first_image := example.first_image) is not None:
            first_image = first_image.to(device=device,dtype=dtype) # (1,3,h,w)
            first_img_latents = vae.encode(first_image.unsqueeze(2)) # vae accept shape (B,C,T,H,W), here B=1,T=1
        else:
            first_img_latents = None
        
        input_size = (example.num_frames, example.height, example.width)
        latent_size = vae.get_latent_size(input_size)
        sample = sample_func(
            model,
            text_encoder,
            z_size=(vae.out_channels, *latent_size),
            window_size=example.auto_regre_chunk_len,
            prompts=[example.prompt],
            first_img_latents=first_img_latents, # (B,C,1,H,W)
            use_predicted_first_img = False,
            txt_guidance_scale = example.txt_guidance_scale,
            img_guidance_scale = example.img_guidance_scale,

            clean_prefix = val_cfgs.clean_prefix,
            clean_prefix_set_t0 = val_cfgs.clean_prefix_set_t0,
            kv_cache_dequeue = kv_cache_dequeue,
            kv_cache_max_seqlen = kv_cache_max_seqlen,
            device = device
        ) # (1, C, T, H, W)
        if enable_kv_cache:
            model.empty_kv_cache()
        sample = vae.decode(sample.to(dtype=dtype))[0] # (C, T, H, W)

        video_name = f"idx{idx}_seed{current_seed}.mp4"
        save_path = os.path.join(save_dir,video_name)
        save_sample(sample.clone(),fps=8,save_path=save_path)

        if writer is not None:
            low, high = (-1,1)
            sample.clamp_(min=low, max=high)
            sample.sub_(low).div_(max(high - low, 1e-5)) # -1 ~ 1 --> 0 ~ 1
            sample = sample.clamp_(0,1).float().cpu()
            sample = sample.unsqueeze(0).permute(0,2,1,3,4) # BCTHW --> BTCHW

            writer.add_video(
                f"validation-{idx}",
                sample,
                global_step = global_step,
                fps=8,
                walltime=None
            )

    if enable_kv_cache:
        model.empty_kv_cache()
    
    
    gc.collect()
    torch.cuda.empty_cache()



def build_validate_examples(examples_or_path,sample_cfgs,print_fn):
    if isinstance(examples_or_path,str):
        examples = load_jsonl(examples_or_path)
    else:
        examples = examples_or_path
    assert isinstance(examples,list) and isinstance(examples[0],dict)

    transforms = torchvision.transforms.Compose(
        [
            video_transforms.ToTensorVideo(), # TCHW, normalize to 0~1
            video_transforms.UCFCenterCropVideo(sample_cfgs.height), # TODO if width != height
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],inplace=True) # To -1 ~ 1
        ]
    )

    print_fn("=="*30 + "use the following validation data"+"=="*30)
    examples_ = []
    for idx, example in enumerate(examples):
        prompt = example.pop("prompt")
        common_quality_prompt = sample_cfgs.get("common_quality_prompt",None)
        if common_quality_prompt:
            prompt = prompt + ', ' + common_quality_prompt
        # NOTE negative prompt is not used as in diffusers. Here we use text_encoder.null (i.e., text_encoder.y_embedder)

        first_image_path = example.pop("first_image",None)
        if first_image_path:
            first_image = torchvision.io.read_image(first_image_path,torchvision.io.ImageReadMode.RGB) # (3,h,w)
            first_image = transforms(first_image.unsqueeze(0)) # (1,3,h,w)
        else:
            first_image = None
        
        data_dict = EasyDict(sample_cfgs.to_dict())
        data_dict.update(EasyDict(dict(
            seed = sample_cfgs.get("seed","random"), # if "random", we will generate a seed according to datetime.now()
            first_image = first_image,
            prompt = prompt,
        )))
        # update other cfgs, if any, e.g., num_frames, seed, guidance_scale
        data_dict.update(example)
        examples_.append(data_dict)

        print_fn(f"idx-{idx}: ")
        for k,v in data_dict.items():
            if k=="first_image": v=first_image_path
            print_fn(f"   {k}:{v}")
    
    return examples_

def merge_args(cfg,args):
    default_cfgs = dict(
        clean_prefix = False,
        clean_prefix_set_t0 = False,
        prefix_perturb_t = -1,
        txt_dropout_proib = 0,
        img_dropout_proib = 0,
        enable_kv_cache = True,

        ### random sample prefix_len = 1 ~ max_L - 1
        prefix_min_len = 1,
        reweight_loss_const_len = 16,
        reweight_loss_per_frame = False,

        ### if auto-regre window size at training & inference time
        # if fix_ar_size =True, disable the above `reweight_loss_const_len` and `reweight_loss_per_frame`
        # and the prefix_len is sampled from [1, 5, 9, ...] (for prefix_min_len=1 and ar_step=4)
        fix_ar_size = False,
        ar_size = 4,
        prefix_sampling_strategy = None, # TODO: e.g., sample short prefix at early training epochs and longer prefix later

        ### dataloader cfgs:
        sampler_seed = None,
    )

    for k, v in default_cfgs.items():
        if k not in cfg:
            cfg[k] = v
    
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    
    assert args.ckpt_path is not None
    cfg.model.from_pretrained = args.ckpt_path

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="./configs/default.py",help="training config")
    parser.add_argument("--ckpt_path",type=str, default="/path/to/ckpt",help="training config")
    parser.add_argument("--outputs",type=str, default=None)
    args = parser.parse_args()

    configs = Config.fromfile(args.config)
    configs = merge_args(configs,args)

    main(configs)
