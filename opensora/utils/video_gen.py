import os
import gc
import time
from tqdm import tqdm
from datetime import datetime
import torch
import torchvision
import torch.distributed as dist
from colossalai.utils import get_current_device,set_seed
from diffusers.schedulers import LCMScheduler
from opensora.datasets import save_sample
from opensora.registry import SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype

from .train_utils import build_progressive_noise
from .debug_utils import envs

@torch.no_grad()
def validation_visualize(model,vae,text_encoder,val_examples,val_cfgs,exp_dir,writer,global_step):
    gc.collect()
    torch.cuda.empty_cache()
    
    device = get_current_device()
    dtype = to_torch_dtype(val_cfgs.dtype)
    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    
    save_dir = os.path.join(exp_dir,"val_samples",f"{global_step}")
    os.makedirs(save_dir,exist_ok=True)

    scheduler = build_module(val_cfgs.scheduler,SCHEDULERS)
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
    additional_kwargs.update(dict(
        progressive_alpha=val_cfgs.get("progressive_alpha",-1)
    ))
    
    for idx,example in enumerate(val_examples):
        current_seed = example.seed
        if current_seed == "random":
            current_seed = int(str(datetime.now().timestamp()).split('.')[-1][:4])

        if (first_image := example.first_image) is not None:
            first_image = first_image.to(device=device,dtype=dtype) # (1,3,h,w)
            cond_frame_latents = vae.encode(first_image.unsqueeze(2)) # vae accept shape (B,C,T,H,W), here B=1,T=1
        else:
            cond_frame_latents = None
        
        # input_size = (example.num_frames, example.height, example.width)
        # latent_size = vae.get_latent_size(input_size)
        assert vae.patch_size[0] == 1
        input_size = (example.auto_regre_chunk_len, example.height, example.width)
        latent_size = vae.get_latent_size(input_size)
    
        samples,time_used,num_gen_frames = sample_func(
            scheduler, 
            model, 
            text_encoder, 
            z_size = (vae.out_channels, *latent_size), 
            prompts = [example.prompt],
            cond_frame_latents=cond_frame_latents, # (B,C,1,H,W)
            ar_steps = example.auto_regre_steps,
            seed = current_seed,
            verbose=True,
            **additional_kwargs
        ) # (1, C, T, H, W)
        fps = num_gen_frames / time_used
        print(f"num_gen_frames={num_gen_frames}, time_used={time_used:.2f}, fps={fps:.2f}")
        vae.micro_batch_size = 16
        sample = vae.decode(samples.to(dtype=dtype))[0] # (C, T, H, W)
        vae.micro_batch_size = None

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
    
    gc.collect()
    torch.cuda.empty_cache()

def denormalize(x, value_range=(-1, 1)):

    low, high = value_range
    x.clamp_(min=low, max=high)
    x.sub_(low).div_(max(high - low, 1e-5))
    x = x.mul(255).add_(0.5).clamp_(0, 255)

    return x

# device = next(model.parameters()).device
def autoregressive_sample_kv_cache(
    scheduler, model, text_encoder, 
    z_size, prompts, cond_frame_latents, ar_steps,
    kv_cache_dequeue, kv_cache_max_seqlen, verbose=True,
    **kwargs
):
    
    # cond_frame_latents: (B, C, T_c, H, W)
    # NOTE: cond_frame_latents output from vae with bf16, here we cast all tensor to fp32 for better accuracy
    # i.e., make sure bf16 is used only inside vae & STDiT model, outside which we all use fp32
    device_dtype = dict(device=cond_frame_latents.device,dtype=torch.float32)
    bsz = len(prompts)
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)
    do_cls_free_guidance = scheduler.cfg_scale > 1.0

    z_predicted = cond_frame_latents.clone().to(**device_dtype)  # (B,C, T_c, H, W)
    
    time_start = time.time()
    num_given_frames = z_predicted.shape[2]
    
    model.register_kv_cache(
        bsz*2 if do_cls_free_guidance else bsz,
        max_seq_len = kv_cache_max_seqlen,
        kv_cache_dequeue = kv_cache_dequeue
    )
    if text_encoder is not None:
        model_kwargs = text_encoder.encode(prompts) # {y,mask}
        y_null = text_encoder.null(bsz) if do_cls_free_guidance else None
    else:
        model_kwargs = {"y":None,"mask":None} 
    
    if do_cls_free_guidance:
        model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)

    model.write_latents_to_cache(
        torch.cat([z_predicted]*2,dim=0) if do_cls_free_guidance else z_predicted,
        **model_kwargs
    )

    generator = torch.Generator(z_predicted.device)
    if seed:=kwargs.get("seed",None):
        generator.manual_seed(seed)

    init_noise = torch.randn(final_size,generator=generator,**device_dtype)
    progressive_alpha = kwargs.get("progressive_alpha",-1)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]
        if progressive_alpha>0: 
            # TODO verify this, check the video gen result is correct
            last_cond = z_predicted[:,:,-1:,:,:]
            tT_bsz = int(scheduler.num_timesteps -1)
            tT_bsz = torch.zeros(size=(bsz,),**device_dtype)
            start_noise = scheduler.q_sample(last_cond,tT_bsz, noise = torch.randn_like(last_cond))
            init_noise_chunk = build_progressive_noise(progressive_alpha, (bsz, *z_size), start_noise)
        

        samples = scheduler.sample_v2(
            model,
            z= init_noise_chunk,
            prompts=prompts,
            device= z_predicted.device,
            model_kwargs = model_kwargs,
            progress_bar = verbose
        ) # (B, C,T_n,H,W)
        
        if envs.DEBUG_KV_CACHE3:
            print(f"<autoregressive_sample>: ar_step={ar_step}: samples={samples[0,0,:,0,0]}, {samples.shape}")
            filename = f"with_kv_cache_denoised_chunk_arstep{ar_step:02d}_BCTHW.pt"
            torch.save(samples,f"{envs.TENSOR_SAVE_DIR}/{filename}")
            # assert ar_step < 2

        model.write_latents_to_cache(
            torch.cat([samples]*2,dim=0) if do_cls_free_guidance else samples,
            **model_kwargs
        )
        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        if verbose: 
            print(f"ar_step={ar_step}: given {predicted_len} frames,  denoise:{samples.shape} --> get:{z_predicted.shape}")


    time_used = time.time() - time_start
    num_gen_frames = z_predicted.shape[2] - num_given_frames

    return z_predicted,time_used,num_gen_frames



def autoregressive_sample(
    scheduler, model, text_encoder, 
    z_size, prompts, cond_frame_latents, ar_steps,
    max_condion_frames, verbose=True,
    **kwargs
):
    # cond_frame_latents: (B, C, T_c, H, W)
    # NOTE: cond_frame_latents output from vae with bf16, here we cast all tensor to fp32 for better accuracy
    # i.e., make sure bf16 is used only inside vae & STDiT model, outside which we all use fp32
    device_dtype = dict(device=cond_frame_latents.device,dtype=torch.float32)
    bsz = len(prompts)
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)
    do_cls_free_guidance = scheduler.cfg_scale > 1.0

    z_predicted = cond_frame_latents.clone().to(**device_dtype)  # (B,C, T_c, H, W)
    
    time_start = time.time()
    num_given_frames = z_predicted.shape[2]
    
    if text_encoder is not None:
        model_kwargs = text_encoder.encode(prompts) # {y,mask}
        y_null = text_encoder.null(bsz) if do_cls_free_guidance else None
    else:
        model_kwargs = {"y":None,"mask":None} 
    
    if do_cls_free_guidance:
        model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)

    generator = torch.Generator(z_predicted.device)
    if seed:=kwargs.get("seed",None):
        generator.manual_seed(seed)
    
    init_noise = torch.randn(final_size,generator=generator,**device_dtype)
    progressive_alpha = kwargs.get("progressive_alpha",-1)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]
        if progressive_alpha > 0: 
            # TODO verify this, check the video gen result is correct
            last_cond = z_predicted[:,:,-1:,:,:]
            tT_bsz = int(scheduler.num_timesteps -1)
            tT_bsz = torch.zeros(size=(bsz,),**device_dtype)
            start_noise = scheduler.q_sample(last_cond,tT_bsz, noise = torch.randn_like(last_cond))
            init_noise_chunk = build_progressive_noise(progressive_alpha, (bsz, *z_size), start_noise)
        
        
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
            assert max_condion_frames + denoise_len == model.max_tpe_len 
        # else:
        z_input_temporal_start = z_predicted.shape[2] - z_cond.shape[2]
        model_kwargs.update({"x_temporal_start":z_input_temporal_start})

        
        model_kwargs.update({"x_cond":z_cond})
        if model.temp_extra_in_channels > 0: # ideally remove this, the model is aware of clean-prefix using timestep emb
            mask_channel = torch.zeros_like(z_input[:,:1,:,:1,:1]) # (B,1,T,1,1)
            mask_channel[:,:,:cond_len,:,:] = 1
            if do_cls_free_guidance:
                mask_channel = torch.cat([mask_channel]*2, dim=0) #
            model_kwargs.update({"mask_channel":mask_channel})

        samples = scheduler.sample_v2(
            model,
            z= z_input,
            prompts=prompts,
            device= z_predicted.device,
            model_kwargs = model_kwargs,
            progress_bar = verbose
        ) # (B, C,T_c+T_n,H,W)
        assert samples.shape[2] == cond_len + denoise_len, f"samples.shape={samples.shape}; cond_len={cond_len}, denoise_len={denoise_len}"
        samples = samples[:,:,cond_len:cond_len+denoise_len,:,:]
        
        if envs.DEBUG_KV_CACHE3:
            print(f"<autoregressive_sample>: ar_step={ar_step}: samples={samples[0,0,:,0,0]}, {samples.shape}")
            filename = f"wo_kv_cache_denoised_chunk_arstep{ar_step:02d}_BCTHW.pt"
            torch.save(samples,f"{envs.TENSOR_SAVE_DIR}/{filename}")
            # assert ar_step < 2

        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        if verbose: 
            print(f"ar_step={ar_step}: given {predicted_len} frames,  denoise:{samples.shape} --> get:{z_predicted.shape}")
        

    time_used = time.time() - time_start
    num_gen_frames = z_predicted.shape[2] - num_given_frames

    return z_predicted,time_used,num_gen_frames