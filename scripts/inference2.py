
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
from opensora.datasets import video_transforms

from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import create_tensorboard_writer,save_training_config
from opensora.utils.misc import (
    to_torch_dtype,
    load_jsonl
)
from opensora.utils.video_gen import validation_visualize
from opensora.utils.debug_utils import envs


def build_validate_examples(examples_or_path,sample_cfgs,print_fn):
    if isinstance(examples_or_path,str):
        examples = load_jsonl(examples_or_path)
    else:
        examples = examples_or_path
    assert isinstance(examples,list) and isinstance(examples[0],dict)

    transforms = torchvision.transforms.Compose(
        [
            video_transforms.ToTensorVideo(), # TCHW, normalize to 0~1
            video_transforms.ResizeCenterCropVideo(sample_cfgs.height), #
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

@torch.no_grad()
def main(cfg):
     # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    if not dist.is_initialized():
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
    
    # backup sample configs:
    _backup_path = os.path.join(exp_dir, f"sampling_cfg_backup.json")
    cfg.update(dict(
        local_time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        utc_time_now = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S"),
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
    vae.eval()
    
    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    
    val_examples = cfg.get("examples",None)
    val_examples = val_examples if val_examples is not None else cfg.examples_json 
    val_examples = build_validate_examples(val_examples,cfg.sample_cfgs,print_fn=logger.info)
    global_step = cfg.ckpt_path.split("global_step")[-1]
    try:
        global_step = int(global_step)
    except:
        assert False, f"global_step={global_step}"
    
    validation_visualize(model,vae,text_encoder,val_examples,cfg,exp_dir,writer=None,global_step=global_step)

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

    default_cfgs = dict(
        dtype = "bf16",
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

def get_exps_info():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="./configs/default.py",help="training config")
    parser.add_argument("--train_config",type=str, default=None)
    parser.add_argument("--ckpt_path",type=str, default=None)
    parser.add_argument("--exp_dir",type=str, default=None)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    configs = Config.fromfile(args.config)
    train_configs = Config.fromfile(args.train_config)
    configs = merge_args(configs,train_configs,args)

    main(configs)

    # from copy import deepcopy
    # exp_list = Config.fromfile("/home/gkf/project/CausalSTDiT/configs/baselines/exps_list.py").exps_list
    # for exp_info in exp_list:
    #     configs_i = deepcopy(configs)
    #     for k,v in exp_info.items():
    #         print(k,v)
        
    #     args.train_config = exp_info.pop("train_config")
    #     args.exp_dir = exp_info.pop("exp_dir")
    #     args.ckpt_path = exp_info.pop("ckpt_path")
    #     configs_i.update(exp_info)
    #     train_configs = Config.fromfile(args.train_config)
    #     configs_i = merge_args(configs_i,train_configs,args)
    #     print("-="*80)
    #     print(f"ckpt_path {args.ckpt_path}")
    #     print(f"exp_dir {args.exp_dir}")
    #     print("-="*80)
    #     main(configs_i)

    
