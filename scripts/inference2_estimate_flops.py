
import argparse
import os
from datetime import timedelta,datetime
import json
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis,flop_count_table

from mmengine.config import Config

from colossalai.utils import get_current_device, set_seed

from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.misc import (
    to_torch_dtype,
    load_jsonl
)
from opensora.utils.debug_utils import envs


def build_txt_input(device,dtype):
    # model_kwargs = {"y":y,"mask":mask}  text_input = build_txt_input(device,dtype)
    bsz = 1
    y = torch.randn(size=(bsz,1,120,4096),device=device,dtype=dtype)
    mask = [1, 1, 1, 1, 1, 1, 1] + [0]*113
    mask = torch.as_tensor(mask)[None,:].to(device)
    print(y.shape,mask.shape,mask)

    return {"y":y,"mask":mask} 



def autoregressive_sample_kv_cache(
    model,
    z_size, cond_frame_latents, text_input,
    ar_steps,kv_cache_max_seqlen,kv_cache_dequeue, verbose=True,
    **kwargs
):
    
    # cond_frame_latents: (B, C, T_c, H, W)
    # NOTE: cond_frame_latents output from vae with bf16, here we cast all tensor to fp32 for better accuracy
    # i.e., make sure bf16 is used only inside vae & STDiT model, outside which we all use fp32
    device_dtype = dict(device=cond_frame_latents.device,dtype=torch.float32)
    bsz = 1
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)
    do_cls_free_guidance = False

    z_predicted = cond_frame_latents.clone().to(**device_dtype)  # (B,C, T_c, H, W)
    
    
    model.register_kv_cache(
        bsz*2 if do_cls_free_guidance else bsz,
        max_seq_len = kv_cache_max_seqlen,
        kv_cache_dequeue = kv_cache_dequeue
    )
    if text_input is not None:
        model_kwargs = text_input
    else:
        model_kwargs = {"y":None,"mask":None} 
    

    model.write_latents_to_cache(
        torch.cat([z_predicted]*2,dim=0) if do_cls_free_guidance else z_predicted,
        **model_kwargs
    )

    all_flops_info = ""
    all_total_flops = 0
    init_noise = torch.randn(final_size,**device_dtype)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]


        # calculate the FLOPs of one denoise step
        timestep_input = torch.zeros(size=(bsz,)) + 100 # (bsz,)
        timestep_input = timestep_input.to(device=init_noise_chunk.device,dtype=torch.long)        
        model_input = (init_noise_chunk,timestep_input)
        if "y" in model_kwargs:
            model_input += (model_kwargs["y"],)
            model_input += (model_kwargs["mask"],)
        flop_analysis_results = FlopCountAnalysis(model,model_input)
        total_flops = flop_analysis_results.total()
        print(total_flops)
        all_total_flops += total_flops
        flops_info  = str(flop_count_table(flop_analysis_results))

        # fake denoised samples
        samples = torch.randn_like(init_noise_chunk)
        
        model.write_latents_to_cache(
            torch.cat([samples]*2,dim=0) if do_cls_free_guidance else samples,
            **model_kwargs
        )
        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        info_ = f"ar_step={ar_step}: given {predicted_len} frames,  denoise:{samples.shape} --> get:{z_predicted.shape}, FLOPs={total_flops}"
        flops_info  = info_ + '\n' + flops_info + '\n'
        if verbose: 
            print(info_)
        
        all_flops_info += '\n' + flops_info + '\n'
    return all_total_flops,all_flops_info


def autoregressive_sample(
    model,
    z_size, cond_frame_latents, text_input,
    ar_steps,max_condion_frames, verbose=True,
    **kwargs
):
    # cond_frame_latents: (B, C, T_c, H, W)
    # NOTE: cond_frame_latents output from vae with bf16, here we cast all tensor to fp32 for better accuracy
    # i.e., make sure bf16 is used only inside vae & STDiT model, outside which we all use fp32
    device_dtype = dict(device=cond_frame_latents.device,dtype=torch.float32)
    bsz = 1
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)
    

    z_predicted = cond_frame_latents.clone().to(**device_dtype)  # (B,C, T_c, H, W)

    if text_input is not None:
        model_kwargs = text_input
    else:
        model_kwargs = {"y":None,"mask":None} 

    all_flops_info = ""
    all_total_flops = 0
    init_noise = torch.randn(final_size,**device_dtype)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]
        
        if predicted_len > max_condion_frames:
            print(" >>> condition_frames dequeue")
            z_cond = z_predicted[:,:,-max_condion_frames:,:,:]
        else:
            # predicted_len <=  max_condion_frames, BUT what if predicted_len+denoise_len > max_model_accpet_len ?
            z_cond = z_predicted
        cond_len = z_cond.shape[2]

            
        z_input = torch.cat([z_cond,init_noise_chunk],dim=2) # (B, C, T_c+T_n, H, W)
        if model.relative_tpe_mode != "cyclic":
            # make sure the temporal position emb not out of range
            assert z_input.shape[2] <= model.max_tpe_len, f'''
            max_condion_frames={max_condion_frames},
            cond_len: {z_cond.shape[2]}, denoise_len: {init_noise_chunk.shape[2]}
            z_input_len = cond_len + denoise_len > model.max_tpe_len = {model.max_tpe_len}
            temporal position embedding (tpe) will out of range !
            '''
            # this happens when (max_condion_frames-first_k_given) % chunk_len !=0, 
            # e.g., max_tpe_len=33, cond: [1,8,9,17,25], chunk_len=8, but we set max_condion_frames=27
        
        if model.relative_tpe_mode is None:
            assert max_condion_frames + denoise_len == model.max_tpe_len, f"{max_condion_frames}+{denoise_len}!={model.max_tpe_len}" 
        # else:
        z_input_temporal_start = z_predicted.shape[2] - z_cond.shape[2]
        model_kwargs.update({"x_temporal_start":z_input_temporal_start})

        
        model_kwargs.update({"x_cond":z_cond})
        if model.temp_extra_in_channels > 0: # ideally remove this, the model is aware of clean-prefix using timestep emb
            mask_channel = torch.zeros_like(z_input[:,:1,:,:1,:1]) # (B,1,T,1,1)
            mask_channel[:,:,:cond_len,:,:] = 1
            model_kwargs.update({"mask_channel":mask_channel})

        
        # calculate the FLOPs of one denoise step
        timestep_input = torch.zeros(size=(bsz,)) + 100 # (bsz,)
        timestep_input = timestep_input.to(device=z_input.device,dtype=torch.long)        
        model_input = (z_input,timestep_input)
        if "y" in model_kwargs:
            model_input += (model_kwargs["y"],)
            model_input += (model_kwargs["mask"],)
        if "mask_channel" in model_kwargs:
            model_input += (model_kwargs["mask_channel"],)
        if "x_temporal_start" in model_kwargs:
            model_input += (model_kwargs["x_temporal_start"],)
        # def forward(self, x, timestep, y, mask=None, mask_channel=None, x_temporal_start=None): 
        # T: 10^12
        # G: 10^9
        # M: 10^6
        # 1,546,102,554,624 = 1.546T 
        # 1545G = 1.545T
        flop_analysis_results = FlopCountAnalysis(model,model_input)
        total_flops = flop_analysis_results.total()
        print(total_flops)
        all_total_flops += total_flops
        # assert False, "details: \n {} \n by_operator:{} \n by_module:{} ".format(
        #     flop_count_table(flop_analysis_results),
        #     flop_analysis_results.by_operator(),
        #     flop_analysis_results.by_module()
        # )
        # TODO: save it into an txt
        flops_info  = str(flop_count_table(flop_analysis_results))

        # fake denoised samples
        samples = torch.randn_like(z_input)
        assert samples.shape[2] == cond_len + denoise_len, f"samples.shape={samples.shape}; cond_len={cond_len}, denoise_len={denoise_len}"
        samples = samples[:,:,cond_len:cond_len+denoise_len,:,:]
        

        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        info_ = f"ar_step={ar_step}: model_input:{z_input.shape[2]}, cond:{z_cond.shape[2]}  denoise:{samples.shape[2]} --> get:{z_predicted.shape}, FLOPs={total_flops}"
        flops_info  = info_ + '\n' + flops_info + '\n'
        if verbose: 
            print(info_)
        
        all_flops_info += '\n' + flops_info + '\n'
    return all_total_flops,all_flops_info

@torch.no_grad()
def main(cfg):
     # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    device = get_current_device() # torch.device("cpu") # 
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
    text_encoder = cfg.get("text_encoder", None)
    if text_encoder is not None:
        text_encoder_output_dim = 4096
        text_encoder_model_max_length = 120
        text_input = build_txt_input(device,dtype)
    else:
        text_encoder_output_dim = 0
        assert cfg.model.caption_channels==0
        text_encoder_model_max_length = 0
        text_input = None
    print(text_encoder_output_dim,text_encoder_model_max_length,text_input is not None)
    vae = build_module(cfg.vae, MODELS)
    bsz=1
    num_given_frames = cfg.get("num_given_frames", 1)
    cond_frame_latents = torch.randn(size=(bsz,4,num_given_frames,32,32),device=device,dtype=dtype)
    
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
    model = model.to(device=device,dtype=dtype)
    model.eval()
    
    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    
    validation_visualize(model,text_input,cond_frame_latents,cfg)

@torch.no_grad()
def validation_visualize(model,text_input,cond_frame_latents,val_cfgs):
    
    assert val_cfgs.get("clean_prefix",True), "TODO add code for non first frame conditioned"
    assert val_cfgs.get("clean_prefix_set_t0",True)
    if enable_kv_cache := val_cfgs.get("enable_kv_cache",False):
        sample_func = autoregressive_sample_kv_cache
        additional_kwargs = dict(
            kv_cache_dequeue = val_cfgs.kv_cache_dequeue,
            kv_cache_max_seqlen = val_cfgs.kv_cache_max_seqlen,
        )
        # `kv_cache_max_seqlen` serves as `max_condion_frames` for sampling w/ kv-cache
    else:
        sample_func = autoregressive_sample
        additional_kwargs = dict(
            max_condion_frames = val_cfgs.max_condion_frames
        )
    
    z_size = (4,8,32,32) # c,chunk_len,h,w = 4,8,32,32
    bsz = 1
    all_total_flops,all_flops_info = sample_func(
        model, 
        z_size, 
        cond_frame_latents=cond_frame_latents, # (B,C,1,H,W)
        text_input = text_input,
        ar_steps = val_cfgs.sample_cfgs.auto_regre_steps,
        verbose=True,
        **additional_kwargs
    ) # (1, C, T, H, W)
    print(f"all_total_flops={all_total_flops}")
    
    save_path = os.path.join(val_cfgs.exp_dir,"all_flops_info.txt")
    with open(save_path,'w') as f:
        f.write(all_flops_info)
    print(f"all_flops_info saved at {save_path}")

def merge_args(cfg,train_cfg,args):
    cfg.update(dict(
        model = train_cfg.model,
        vae = train_cfg.vae,
        clean_prefix = train_cfg.clean_prefix,
        clean_prefix_set_t0 = train_cfg.clean_prefix_set_t0,
    ))
    if "prefix_perturb_t" in cfg.keys():
        # maybe change the prefix_perturb_t at test time
        pass
    else:
        cfg.update(prefix_perturb_t = train_cfg.prefix_perturb_t)

    if "text_encoder" in cfg.keys():
        # maybe change the configs of text_encoder at test time
        pass
    else:
        cfg.update(text_encoder = train_cfg.text_encoder)
    
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
    
    # update model config at inference time:
    # e.g., train w/ seq_parallel and inference w/o seq_parallel
    # e.g., enable_sequence_parallelism, enable_flashattn, etc.
    from copy import deepcopy
    model_kwargs = deepcopy(cfg.model)
    for k, v in cfg.items():
        for k_model in cfg.model.keys():
            if k==k_model:
                model_kwargs.update({k_model:v})
    cfg.update(model=model_kwargs)

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

    
