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
from opensora.utils.video_gen import validation_visualize

def main(cfg):
    # ======================================================
    # 1. args & cfg
    # ======================================================
    assert cfg.fix_ar_size, "refer to _backup/train_backup_before_remove_reweight_loss.py"
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
    if not coordinator.is_master():
        logger = create_logger(None)
        writer = None
    else:
        os.makedirs(exp_dir,exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")
        logger.info(f"Training configuration:\n {pformat(cfg._cfg_dict)}")
        _backup_path = save_training_config(cfg._cfg_dict,exp_dir)
        logger.info(f"Backup training config at {_backup_path}")
        
        writer = create_tensorboard_writer(exp_dir)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    train_data_cfg = deepcopy(cfg.train_data_cfg)
    dataset = build_module(train_data_cfg, DATASETS, print_fn = logger.info)  # VideoTextDatasetFromJson
    logger.info(f"Dataset `{train_data_cfg.type}` is built, with {len(dataset)} videos.")
    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed = cfg.sampler_seed,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    logger.info(f"dataloader is built, with dataloader.sampler.seed={dataloader.sampler.seed}")


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
    input_size = (dataset.n_sample_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
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

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )
    if coordinator.is_master():
        model_structure_path = os.path.join(exp_dir,"model_structure.log")
        with open(model_structure_path,'w') as f:
            f.write(str(model))
        logger.info(f"model_structure saved at {model_structure_path}")
    

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = global_step = 0
    
    # 6.1. resume training
    if cfg.resume_from_ckpt is not None:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from_ckpt}")
        ret = load(
            booster,
            model,
            ema,
            optimizer,
            lr_scheduler,
            cfg.resume_from_ckpt,
        )
        if not cfg.start_from_scratch: # TODO: gkf: figure out what's this 
            start_epoch, start_step, sampler_start_idx = ret
        logger.info(f"Loaded checkpoint at epoch {start_epoch} step {start_step}")
        global_step += resumed_global_step
        assert False, '''
        TODO: check whether the resumed ckpt & start_epoch, start_step, sampler_start_idx are correct
        '''

    num_steps_per_epoch = len(dataloader) // cfg.accumulation_steps
    total_batch_size = (cfg.batch_size * dist.get_world_size() // cfg.sp_size) * cfg.accumulation_steps 
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")
    logger.info(f"Dataset contains {len(dataset)} videos")
    logger.info(f"total batch size (w.r.t. per optimization step): {total_batch_size}")

    dataloader.sampler.set_start_index(sampler_start_idx)
    model_sharding(ema)
    
    # 6.2. build validation examples & validate before train
    val_cfgs = cfg.validation_configs
    val_cfgs.update(dict(
        clean_prefix = cfg.clean_prefix,
        clean_prefix_set_t0 = cfg.clean_prefix_set_t0,
        dtype = cfg.dtype,
    ))
    val_examples = val_cfgs.get("examples",None)
    val_examples = val_examples if val_examples is not None else val_cfgs.examples_json 
    val_examples = build_validate_examples(val_examples,val_cfgs.sample_cfgs,print_fn=logger.info)
    if cfg.validate_before_train and coordinator.is_master():
        model.eval()
        validation_visualize(model.module,vae,text_encoder,val_examples,val_cfgs,exp_dir,writer,global_step)
        model.train()

    # 6.3. training loop

    for epoch in range(start_epoch, cfg.epochs):
        
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}... num_steps_per_epoch = {num_steps_per_epoch} ")
        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch, # we have set `initial=start_step`, so total should NOT be `num_steps_per_epoch-start_step`
        ) as pbar:
            
            running_loss = torch.tensor(0.0,device=get_current_device())
            loss_accu_step = 0
            for step, batch in enumerate(dataloader_iter):
                is_last_step = (step == len(dataloader_iter) - 1)
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")
                # Visual and text encoding
                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
                    if text_encoder is not None:
                        model_kwargs = text_encoder.encode(y) # {y,y_mask}
                    else:
                        model_kwargs = {"y":None,"mask":None}
                    
                bsz = x.shape[0]
                actual_length = batch["actual_length"]
                if cfg.clean_prefix:
                    '''e.g.,
                                [frame prompt    | frame to denoise| padding    ]
                                [clean_prefix    | noisy latents   | padding    ]
                    latents:    [z0,z1,z2,...,z9,| z18,z19,...,z57 | 0,0,0,...,0]
                    mask_channel[1,1,1,...     1,| 0,0,0,...     0,| 0,0,0,...,0]
                    loss_mask   [0,0,0,...,    0,| 1, 1, ...,    1,| 0,0,0,...,0]

                    '''
                    assert actual_length is not None
                    assert cfg.prefix_min_len < (min_act_L := (actual_length.min().item())), "mask sure use condition_th to filter data"
                    assert cfg.img_dropout_prob == 0 or cfg.img_dropout_prob==0.0,"TODO: refer to _backup/train_backup_before_remove_reweight_loss.py"

                    pL = prefix_len_sampler.random_choose(min_act_L)
                    loss_mask = torch.zeros_like(x) # (bsz,c,f,h,w)
                    mask_channel = torch.zeros_like(loss_mask[:,0:1,:,:1,:1]) # (bsz,1,f,1,1)
                    denoise_len = min(cfg.ar_size,min_act_L)
                    loss_mask[:,:,pL:pL+denoise_len,:,:] = 1
                    mask_channel[:,:,:pL,:,:] = 1
                    
                    model_kwargs.update(dict(
                        mask_channel = mask_channel, # (bsz,1,f,1,1)
                        loss_mask = loss_mask, # (bsz,c,f,h,w)
                        prefix_perturb_t = cfg.prefix_perturb_t,
                    ))
                elif actual_length is not None:
                    # if clean_prefix is turned off, but some sample in batch maybe shorter than `f`
                    loss_mask = torch.zeros_like(x) # (bsz,c,f,h,w)
                    for b,act_L in enumerate(actual_length.tolist()):
                        loss_mask[b,:,:act_L,:,:] = 1
                    model_kwargs.update(dict(
                        loss_mask = loss_mask, # (bsz,c,f,h,w)
                    ))
                    if cfg.fix_ar_size: assert False, "TODO"
                else:
                    loss_mask = None

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                if cfg.clean_prefix and cfg.clean_prefix_set_t0:
                    # mask_channel: (bsz,1,f,1,1)
                    num_frames = x.shape[2]
                    t_input = t[:,None].repeat(1,num_frames) # (bsz,f)
                    # NOTE we do not modift `t` inplace, because `p_sample_loop` only accepts `t` with shape (bsz,)
                    t_input[:,:pL] = 0
                    model_kwargs.update(dict(t_input=t_input))

                if loss_mask is not None:
                    if cfg.scheduler.type == "clean_prefix_iddpm":
                        loss_func = scheduler.training_losses_clean_prefix
                    else:
                        loss_func = scheduler.training_losses_with_mask # TODO: in opensora-v1.1, GaussianDiffusion.training_losses supports input mask
                else:
                    loss_func = scheduler.training_losses
                    model_kwargs.pop("loss_mask",None)
                
                if (alpha:=cfg.scheduler.progressive_alpha) > 0:
                    _noise = torch.randn_like(x) # (bsz,c,f,h,w)
                    prev_noise = _noise[:,:,0:1,:,:] # (bsz,c,1,h,w)
                    progressive_noises = [prev_noise]
                    for i in range(1,x.shape[2]):
                        new_noise = (alpha / math.sqrt(1+alpha**2)) * prev_noise + (1/math.sqrt(1+alpha**2)) * _noise[:,:,i:i+1,:,:]
                        progressive_noises.append(new_noise)
                        prev_noise = new_noise
                    progressive_noises = torch.cat(progressive_noises,dim=2) # (b,c,f,h,w)
                    custom_noise = progressive_noises
                else:
                    custom_noise = None
                
                loss_dict = loss_func(model, x, t, model_kwargs, noise=custom_noise)

                # Backward
                loss = loss_dict["loss"].mean() / cfg.accumulation_steps
                running_loss.add_(loss.detach()) # for log
                loss_accu_step += 1 # for log
                booster.backward(loss=loss, optimizer=optimizer)

                # Update
                is_update = (step + 1) % cfg.accumulation_steps == 0 or is_last_step
                if is_update:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update EMA
                    update_ema(ema, model.module, optimizer=optimizer)

                    global_step += 1
                    pbar.update(1)
                
                # Log loss values:
                if global_step % cfg.log_every_step == 0 and is_update:

                    
                    all_reduce_mean(running_loss) # reduce_mean across all gpus
                    avg_loss = running_loss.item() / loss_accu_step  # avg across log_every_step and n gpus
                    if coordinator.is_master():
                        pbar.set_postfix({"loss": avg_loss, "global_step": global_step})
                        writer.add_scalar("loss", avg_loss, global_step)
                        logger.info(f"global_step={global_step}: interval_avg_loss = {avg_loss}")
                    running_loss.zero_()
                    loss_accu_step = 0


                # Save checkpoint
                if cfg.ckpt_every_step > 0 and global_step % cfg.ckpt_every_step == 0 and is_update:
                    not_save_optimizer = cfg.get("not_save_optimizer",False)
                    save(
                        booster,
                        model,
                        ema,
                        None if not_save_optimizer else optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        ema_shape_dict,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )
                
                if (cfg.validation_every_step > 0) and (global_step % cfg.validation_every_step==0) and is_update:
                    if coordinator.is_master():
                        model.eval()
                        validation_visualize(model.module,vae,text_encoder,val_examples,val_cfgs,exp_dir,writer,global_step)
                        model.train()
                        assert not envs.DEBUG_WITHOUT_LOAD_PRETRAINED # incase we forgot to turn this off

                    dist.barrier()

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

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



def reweight_loss_mask(loss_mask,const_len):
    # loss_maskL (b,c,f,h,w)
    bsz,c,f,h,w = loss_mask.shape
    window_size = const_len
    loss_weights = torch.zeros_like(loss_mask)
    for b in range(bsz):
        n_pred = loss_mask[b,0,:,0,0].sum()
        n_prefix = torch.argmax(loss_mask[b,0,:,0,0])
        loss_weights[b,:,n_prefix:n_prefix+window_size,:,:] = 1
        if (n_decay:=(int(n_pred) - window_size)) > 0:
            decay_weights = torch.linspace(0.9,0.01,steps=n_decay)
            decay_weights = decay_weights[None,:,None,None].repeat(c,1,h,w).to(loss_weights.device)
            loss_weights[b,:,n_prefix+window_size:n_prefix+window_size+n_decay,:,:] = decay_weights # (c,n_decay,h,w)
    
    return loss_weights

def merge_args(cfg,args):
    default_cfgs = dict(
        clean_prefix = True,
        clean_prefix_set_t0 = True,
        prefix_perturb_t = -1,
        txt_dropout_prob = 0,
        img_dropout_prob = 0,
        enable_kv_cache = True,

        ### random sample prefix_len = 1 ~ max_L - 1
        prefix_min_len = 1,
        ar_size = 4,
        prefix_sampling_strategy = None, # TODO: e.g., sample short prefix at early training epochs and longer prefix later

        ### dataloader cfgs:
        sampler_seed = None,

        ### deprecated_cfgs
        fix_ar_size= True
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
    parser.add_argument("--outputs",type=str, default=None)
    args = parser.parse_args()

    configs = Config.fromfile(args.config)
    configs = merge_args(configs,args)

    main(configs)

    '''
    bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/baselines/exp4_partialcausal_attn_cyclic_tpe33.py \
        9686 0
    '''
