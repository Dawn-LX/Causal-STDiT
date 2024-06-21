from functools import partial
import math
import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .speed import SpeeDiffusion

def build_progressive_noise(alpha,noise):
    # noise (bsz,c,f,h,w)
    if alpha > 0:
        prev_noise = noise[:,:,0:1,:,:] # (bsz,c,1,h,w)
        progressive_noises = [prev_noise]
        for i in range(1,noise.shape[2]):
            new_noise = (alpha / math.sqrt(1+alpha**2)) * prev_noise + (1/math.sqrt(1+alpha**2)) * noise[:,:,1:i+1,:,:]
            progressive_noises.append(new_noise)
            prev_noise = new_noise
        progressive_noises = torch.cat(progressive_noises,dim=2) # (b,c,f,h,w)
        noise = progressive_noises
    return noise

@SCHEDULERS.register_module("iddpm")
class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
        cfg_channel=None,
        progressive_alpha = -1,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale
        self.cfg_channel = cfg_channel
        self.progressive_alpha = progressive_alpha

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        **kwargs
    ):
        n = len(prompts)
        if self.progressive_alpha > 0:
            z = build_progressive_noise(self.progressive_alpha,z)
        
        z = torch.cat([z, z], 0)
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale, cfg_channel=self.cfg_channel)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            mask=mask,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples

    def training_losses_with_mask(self, model, *args, **kwargs):
        return self._training_losses_with_mask(self._wrap_model(model), *args, **kwargs)
    
    def _training_losses_with_mask(self, model, x_start, t, model_kwargs=None, noise=None):
        '''
        add loss mask for training samples with variable lenghts (with zero padding)
        this is for turn-off clean_prefix, but turn on is_causal
        '''
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        loss_mask = model_kwargs.pop("loss_mask",None)
        
        terms = {}

        '''TODO:
        if self.loss_type == gd.LossType.KL or self.loss_type == gd.LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == gd.LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        '''

        assert self.loss_type == gd.LossType.MSE or self.loss_type == gd.LossType.RESCALED_MSE
        model_output = model(x_t, t, **model_kwargs)

        if self.model_var_type in [
            gd.ModelVarType.LEARNED,
            gd.ModelVarType.LEARNED_RANGE,
        ]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            vb_loss = self._vb_terms_bpd_keep_dim(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            if self.loss_type == gd.LossType.RESCALED_MSE:
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                terms["vb"] *= self.num_timesteps / 1000.0
            
            if loss_mask is not None:
                terms["vb"] = sum_flat(vb_loss * loss_mask) / sum_flat(loss_mask)
            else:
                terms["vb"] = gd.mean_flat(vb_loss)


        target = {
            gd.ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
            gd.ModelMeanType.START_X: x_start,
            gd.ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape
        if loss_mask is not None:
            mse_loss = (target - model_output) ** 2
            terms["mse"] = sum_flat(mse_loss * loss_mask) / sum_flat(loss_mask)
        else:
            
            terms["mse"] = gd.mean_flat((target - model_output) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]
        
        return terms

def forward_with_cfg(model, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    if "x_mask" in kwargs and kwargs["x_mask"] is not None:
        if len(kwargs["x_mask"]) != len(x):
            kwargs["x_mask"] = torch.cat([kwargs["x_mask"], kwargs["x_mask"]], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs) # (2b,2c,f,h,w)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    if cfg_channel is None:
        cfg_channel = model_out.shape[1] // 2
    eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:] # (2b,c,f,h,w)
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0) # (b,c,f,h,w)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) # (b,c,f,h,w)
    eps = torch.cat([half_eps, half_eps], dim=0) # (2b,c,f,h,w)
    return torch.cat([eps, rest], dim=1) # (2b,2c,f,h,w)

@SCHEDULERS.register_module("clean_prefix_iddpm")
class CleanPrefixIDDPM(IDDPM):
    def training_losses_clean_prefix(self, model, *args, **kwargs):
        return self._training_losses_clean_prefix(self._wrap_model( model), *args, **kwargs)
    
    def _training_losses_clean_prefix(self,model,x_start,t,model_kwargs=None,noise=None):
        '''e.g.,
                    [frame prompt    | frame to denoise| padding    ]
                    [clean_prefix    | noisy latents   | padding    ]
        latents:    [z0,z1,z2,...,z9,| z18,z19,...,z57 | 0,0,0,...,0]
        mask_channel[1,1,1,...     1,| 0,0,0,...     0,| 0,0,0,...,0]
        loss_mask   [0,0,0,...,    0,| 1, 1, ...,    1,| 0,0,0,...,0]

        '''

        if model_kwargs is None:
            model_kwargs = {}
            assert False, "should not be here"
        
        loss_mask = model_kwargs.pop("loss_mask",None) # (bsz,c,f,h,w)
        mask_channel = model_kwargs.get("mask_channel",None) # (bsz,1,f,1,1); NOTE do not pop this 
        assert loss_mask is not None

        if noise is None:
            noise = torch.randn_like(x_start)
        x_clean = x_start.clone().detach()
        x_t = self.q_sample(x_start, t, noise=noise)

        if mask_channel is not None:
            x_clean_mask = mask_channel.expand_as(x_t).type(torch.bool)
            x_t = torch.where(x_clean_mask,x_clean,x_t) # 1 for use `x_clean`, 0 for use `x_t`
            pp_t= model_kwargs.pop("prefix_perturb_t",-1)
            if pp_t >= 0 :
                pp_t = torch.zeros_like(t) + pp_t # (bsz,)
                perturb_noise = torch.randn_like(x_clean)
                x_clean_perturbed = self.q_sample(x_clean,pp_t, noise=perturb_noise)
                x_t = torch.where(x_clean_mask,x_clean_perturbed,x_t)


        '''TODO:
        if self.loss_type == gd.LossType.KL or self.loss_type == gd.LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == gd.LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        '''

        assert self.loss_type == gd.LossType.MSE or self.loss_type == gd.LossType.RESCALED_MSE

        t_input = model_kwargs.pop("t_input",t) # use custom timestep input (with shape (bsz,f)) if `cfg.clean_prefix_set_t0=True`
        model_output = model(x_t, t_input, **model_kwargs)

        terms = {}
        if self.model_var_type in [
            gd.ModelVarType.LEARNED,
            gd.ModelVarType.LEARNED_RANGE,
        ]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            vb_loss = self._vb_terms_bpd_keep_dim(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            if self.loss_type == gd.LossType.RESCALED_MSE:
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                terms["vb"] *= self.num_timesteps / 1000.0
            
            if loss_mask is not None:
                terms["vb"] = sum_flat(vb_loss * loss_mask) / sum_flat(loss_mask)
            else:
                terms["vb"] = gd.mean_flat(vb_loss)


        target = {
            gd.ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
            gd.ModelMeanType.START_X: x_start,
            gd.ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape

        if loss_mask is not None:
            mse_loss = (target - model_output) ** 2
            terms["mse"] = sum_flat(mse_loss * loss_mask) / sum_flat(loss_mask)
        else:
            
            terms["mse"] = gd.mean_flat((target - model_output) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]
        
        return terms
    
    @torch.no_grad()
    def sample(
        self,
        model,
        text_encoder,
        z_size, # (c,f,h,w)
        window_size, # span (chunk_len) for each auto-regression step
        prompts, # list[str]
        first_img_latents, # (bsz,c,1,h,w)
        use_predicted_first_img = False,
        clean_prefix=True,
        clean_prefix_set_t0 = True,
        txt_guidance_scale = 6.0,
        img_guidance_scale = 1.0,
        device = torch.device("cuda"),
        progress_bar = True,
    ):
        '''
        consider classifier_free_guidance for both txt and img
        refer to https://github.com/TIGER-AI-Lab/ConsistI2V/blob/d40d64b4c8f005ee8a4915528df705e7d586ea9a/consisti2v/pipelines/pipeline_conditional_animation.py#L341
        '''
        bsz,num_frames = len(prompts)
        if first_img_latents is None:
            _first_img_latents = model.bov_token[None,:,None,:,:].repeat(bsz,1,1,1,1) # (bsz,c,1,h,w)
            _c,_f,_h,_w = z_size
            _z_size = (_c,window_size+1,_h,_w)
            first_window_pred = self.sample(
                model,
                text_encoder,
                _z_size,
                window_size,
                prompts,
                _first_img_latents,
                use_predicted_first_img=True,
                clean_prefix=clean_prefix,
                clean_prefix_set_t0=clean_prefix_set_t0,
                txt_guidance_scale=txt_guidance_scale,
                img_guidance_scale=img_guidance_scale,
                device=device,
            )
            first_img_latents = first_window_pred[:,:,1,:,:] # (bsz,c,h,w)

        if first_img_latents.ndim==4:
            first_img_latents = first_img_latents.unsqueeze(2) # (bsz,c,1,h,w)
        
        # two guidance mode: txt & txt+img
        cls_free_guidance = None
        if txt_guidance_scale > 1.0:
            cls_free_guidance = "text"
        if img_guidance_scale > 1.0:
            cls_free_guidance = "both"
            assert txt_guidance_scale > 1.0, "turn on txt guidance when using img guidance"
        
        # 1. build txt condition
        model_kwargs = text_encoder.encode(prompts)
        y_null = text_encoder.null(bsz) if cls_free_guidance else None
        if cls_free_guidance == "text":
            model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)
        elif cls_free_guidance == "both":
            model_kwargs["y"] = torch.cat([y_null,y_null,model_kwargs["y"]], dim=0)
            # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
        
        model_kwargs.update(dict(
            clean_prefix = clean_prefix,
            clean_prefix_set_t0 = clean_prefix_set_t0
        ))

        # 2. get auto-regression steps
        first_k_gievn = first_img_latents.shape[2]
        assert first_k_gievn < num_frames and num_frames < self.MAX_AUTO_REGRESSION_LEN # remove the second term when enable kv_cache_dequeue
        num_gen = num_frames - first_k_gievn

        window_sizes = []
        accumulate_frames = first_k_gievn
        while accumulate_frames < num_frames:
            ws = min(window_size, num_frames - accumulate_frames)
            window_sizes.append(ws)
            accumulate_frames += ws
        assert accumulate_frames == num_frames
        auto_regre_steps = len(window_sizes)
        if progress_bar: print(f"window_size: {window_sizes}")

        # 3. build noise given first image
        noise = torch.randn(bsz, *z_size, device = device) # (bsz,c,f,h,w)
        if self.progressive_alpha > 0:
            noise = build_progressive_noise(self.progressive_alpha,noise)
        
        z_predicted = first_img_latents.clone() # (bsz,c,1,h,w)
        for ar_step in range(auto_regre_steps):
            window_size = window_sizes[ar_step]
            predict_start_id = z_predicted.shape[2]
            noise_prefix = noise[:,:,:predict_start_id,:,:] # (bsz,c,f',h,w)
            noise_window = noise[:,:,predict_start_id:predict_start_id+window_size,:,:] # (bsz,c,ws,h,w)

            if clean_prefix:
                mask_channel = torch.zeros_like(noise)[:,0:1,:predict_start_id+window_size,:1,:1] # (b,1,f'+ws,1,1)
                mask_channel[:,:,:predict_start_id,:,:] = 1
                if cls_free_guidance == "text":
                    mask_channel = torch.cat([mask_channel]*2, dim=0) # (2b,1,f'+ws,1,1)
                elif cls_free_guidance == "both":
                    mask_channel = torch.cat([torch.zeros_like(mask_channel),mask_channel,mask_channel],dim=0) # (3b,1,f'+ws,1,1)
                    # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
                model_kwargs.update(dict(mask_channel=mask_channel))
            
            z_input = torch.cat([noise_prefix,noise_window],dim=2) # (b,c,f'+ws,h,w)
            if progress_bar:
                print(f"ar_step={ar_step}, prefix [0:{z_predicted.shape[2]}); denoising [{predict_start_id}:{predict_start_id+window_size})")
            forward = partial(
                forward_with_cfg2,
                model,
                z_predicted=z_predicted,
                clean_prefix = clean_prefix,
                clean_prefix_set_t0 = clean_prefix_set_t0,
                txt_guidance_scale = txt_guidance_scale,
                img_guidance_scale = img_guidance_scale
            )
            sample = self.p_sample_loop(
                forward,
                z_input.shape, # (bsz,c,f'+ws,h,w); we will duplicate bsz for cls_free_guidance inside `forward_with_cfg2`
                z_input,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=progress_bar,
                device=device
            ) # (bsz,c,f'+ws,h,w)

            # extend z_predicted
            predicted_window = sample[:,:,predict_start_id:,:,:] # (bsz,c,ws,h,w)
            z_predicted = torch.cat([z_predicted,predicted_window],dim=2)

            if use_predicted_first_img and ar_step == 0:
                z_predicted = sample # (bsz,c,1+ws,h,w)
                assert predict_start_id == 1
                assert z_predicted.shape[2] == 1+ws
        
        if progress_bar: print(f"after autoregression, z_predicted.shape={z_predicted.shape}")

        return z_predicted

    @torch.no_grad()
    def sample_with_kv_cache(
        self,
        model,
        text_encoder,
        z_size, # (c,f,h,w)
        window_size, # span (chunk_len) for each auto-regression step
        prompts, # list[str]
        first_img_latents, # (bsz,c,1,h,w)
        use_predicted_first_img = False,
        clean_prefix=True,
        clean_prefix_set_t0 = True,
        txt_guidance_scale = 6.0,
        img_guidance_scale = 1.0,
        device = torch.device("cuda"),
        progress_bar = True,
        kv_cache_max_seqlen = None,
        kv_cache_dequeue = False,
    ):
        assert model.is_causal

        bsz, num_frames = len(prompts),z_size[1]
        if first_img_latents is None:
            pass
            # TODO copy code from self.sample
        
        if first_img_latents.ndim ==4:
            first_img_latents = first_img_latents.unsqueeze(2) # (bsz,c,1,h,w)
        
        # two guidance mode: txt & txt+img
        cls_free_guidance = None
        if txt_guidance_scale > 1.0:
            cls_free_guidance = "text"
        if img_guidance_scale > 1.0:
            cls_free_guidance = "both"
            assert txt_guidance_scale > 1.0, "turn on txt guidance when using img guidance"
        
        assert clean_prefix and clean_prefix_set_t0, '''
            in the denoise loop, each timestep uses the same kv-cache,
            so we have to make sure for each timestep, the clean_prefix part of model_input is same, including timestep_embedding, mask_channel
        '''
        assert cls_free_guidance != "both", '''TODO
            if considering cls_free_guidance == "both", then we duplicate 3 batch, 
            where one of them is img-unconditional, it should be:
            1) prefix is all noise
            2) maskchannel is all 0
            3) timestep embedding is not t0's embedding, it is the timestep embedding of current denoise step
            1) & 2) are fine, but for 3), if we use 3), then each timestep in the denoising loop has different kv-cache.
        '''
        
        # 1. build txt condition
        model_kwargs = text_encoder.encode(prompts)
        y_null = text_encoder.null(bsz) if cls_free_guidance else None
        if cls_free_guidance == "text":
            model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)
            bsz_dup = bsz*2
        elif cls_free_guidance == "both":
            model_kwargs["y"] = torch.cat([y_null,y_null,model_kwargs["y"]], dim=0)
            # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
            bsz_dup = bsz*3
        else:
            bsz_dup = bsz
        
        # 2. get auto-regression steps
        first_k_gievn = first_img_latents.shape[2]
        assert first_k_gievn < num_frames and num_frames < self.MAX_AUTO_REGRESSION_LEN # remove the second term when enable kv_cache_dequeue
        num_gen = num_frames - first_k_gievn

        window_sizes = []
        accumulate_frames = first_k_gievn
        while accumulate_frames < num_frames:
            ws = min(window_size, num_frames - accumulate_frames)
            window_sizes.append(ws)
            accumulate_frames += ws
        assert accumulate_frames == num_frames
        auto_regre_steps = len(window_sizes)
        if progress_bar: print(f"window_size: {window_sizes}") 

        if kv_cache_max_seqlen is None:
            kv_cache_max_seqlen = num_frames
        if kv_cache_dequeue:
            # TODO: remove this line, and set kv_cache_dequeue always True and max_cache is directly `kv_cache_max_seqlen`
            kv_cache_max_seqlen = kv_cache_max_seqlen - window_size
        
        model.pre_allocate_kv_cache(bsz_dup,kv_cache_max_seqlen,kv_cache_dequeue)

        # 3. build noise given first image
        noise = torch.randn(bsz, *z_size, device=device)
        if self.progressive_alpha > 0:
            noise = build_progressive_noise(self.progressive_alpha,noise)
        
        # 4. start auto-regressive sample
        # 4.1 build noise
        z_predicted = first_img_latents.clone() # (bsz,c,1,h,w)
        predict_start_id = z_predicted.shape[2] # i.e., 1 for only given the first frame

        # 4.2 wirte kv-cache for first frame
        model_kwargs.update((dict(start_id = 0)))
        write_kv_cache(model, z_predicted, model_kwargs, cls_free_guidance,
            noise = noise[:,:,0:predict_start_id,:,:] if cls_free_guidance == "both" else None
        )
        # 4.3 auto-regression loop
        for ar_step in range(auto_regre_steps):
            ws = window_sizes[ar_step]
            z_input = noise[:,:,predict_start_id:predict_start_id+ws,:,:] # (bsz,c,ws,h,w)
            '''e.g., for first_k_given=1, ws=8
            ar_step=0 : input [1:9],   predict_start_id = 1, input_seqlen = 8
            ar_step=1 : input [9:17],  predict_start_id = 9, input_seqlen = 8
            ar_step=2 : input [17:25], predict_start_id = 17, input_seqlen = 8
            '''
            if progress_bar:
                print(f"ar_step={ar_step}, prefix [0:{z_predicted.shape[2]}); denoising [{predict_start_id}:{predict_start_id+ws})")
            
            forward = partial(forward_with_cfg2_kv_cache, model,
                txt_guidance_scale = txt_guidance_scale,
                img_guidance_scale = img_guidance_scale
            )
            model_kwargs.update(dict(start_id = predict_start_id))
            samples = self.p_sample_loop(
                forward,
                z_input.shape, # (bsz,c,f'+ws,h,w); we will duplicate bsz for cls_free_guidance inside `forward_with_cfg2`
                z_input,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=progress_bar,
                device=device
            ) # (bsz,c,ws,h,w)
            '''NOTE
            It's diffuclt to get exactly the same frame results as sample w/o kv-cache (even we use the same seed)
            我们很难获得与 sample w/o kv_cache 完全一致的结果（即使seed相同）
            因为 p_sample 中 x_{t-1} = mean + var * noise noise 是按照 z_input.shape 取的 randn， 
            for w/ kv_cache, z_input.shape only include current window
            for w/o kv_cache, z_input.shape include all previous predictions
            '''

            # only record clean sample's kv-cache, and thus, mask_channel is all zeros, timestep emb also use t0's emb
            # so mask_channel & timestep_emb are not fed into model_kwargs, model_kwargs: {y, y_mask, start_id}
            write_kv_cache(model, samples, model_kwargs, cls_free_guidance,
                noise = noise[:,:,predict_start_id:predict_start_id+ws,:,:] if cls_free_guidance == "both" else None
            )

            # extend z_predicted
            z_predicted = torch.cat([z_predicted,samples],dim=2) # 

            if use_predicted_first_img and ar_step==0:
                assert False, "for sample w/ kv-cache, we don't have predicted_first_img"

            predict_start_id += ws
        
        return z_predicted

def forward_with_cfg2(
    model,
    z,              # (b,c,f'+ws,h,w)
    timestep,       # (b,)
    z_predicted,    # (b,c,f',h,w)
    clean_prefix,
    clean_prefix_set_t0,
    txt_guidance_scale,
    img_guidance_scale,
    **model_kwargs # y, mask, mask_channel
):
    '''
    modifications:
        - add cls_free_guidance for txt & img
        - modify z & timestep for clean_prefix & clean_prefix_set_t0
    '''

    # two guidance mode: txt & txt+img
    cls_free_guidance = None
    if txt_guidance_scale > 1.0:
        cls_free_guidance = "text"
    if img_guidance_scale > 1.0:
        cls_free_guidance = "both"
        assert txt_guidance_scale > 1.0, "turn on txt guidance when using img guidance"
    
    n_frames_input = z.shape[2] # f' + window_size
    predict_start_id = z_predicted.shape[2] # f'
    if clean_prefix:
        if cls_free_guidance == "both":
            noisy_z_backup = z.clone() # ?? here we use  t-level noise, But in ConsistI2V,  they use T-level noise (for both training and inference)
            # TODO
        # set clean_prefix
        z[:,:,:predict_start_id,:,:] = z_predicted
    
    t_input = timestep
    if clean_prefix_set_t0:
        assert clean_prefix
        # set t=0 for each clean prefix
        t_input = timestep[:,None,].repeat(1,n_frames_input) # (bsz,f'+ws)
        t_input[:, :predict_start_id] = 0

    if cls_free_guidance == "text":
        z = torch.cat([z]*2, dim=0)
        t_input = torch.cat([t_input]*2, dim=0)
    elif cls_free_guidance == "both":
        z = torch.cat([noisy_z_backup,z,z],dim=0)
        t_input_raw = timestep[:,None].repeat(1, n_frames_input) if clean_prefix_set_t0 else timestep
        t_input = torch.cat([t_input_raw,t_input,t_input], dim=0) # (bsz*3,) or (bsz*3, f) 
        # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
    
    model_out = model.forward(z, t_input, **model_kwargs) # (b,c*2,f'+ws,h,w)

    # split prediction of mean and variance
    out_channel = model_out.shape[1] //2
    noise_pred, var_pred = model_out[:, :out_channel], model_out[:, out_channel:] # (b,c,f'+ws,h,w)

    # perform guidance (only for mean), we do not perform guidance for variance 
    if cls_free_guidance == "text":
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2,dim=0)
        noise_pred = noise_pred_uncond + txt_guidance_scale * (noise_pred_text - noise_pred_uncond) # (b,c,f'+ws,h,w)
        _,var_pred = var_pred.chunk(2,dim=0) # (b,c,f'+ws,h,w)
        
    elif cls_free_guidance == "both":
        # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
        noise_pred_uncond, noise_pred_img, noise_pred_both = noise_pred.chunk(3,dim=0)
        noise_pred = noise_pred_uncond \
            + img_guidance_scale * (noise_pred_img - noise_pred_uncond) \
            + txt_guidance_scale * (noise_pred_both - noise_pred_img)
        _,_,var_pred = var_pred.chunk(3,dim=0)

    model_out = torch.cat([noise_pred,var_pred],dim=1) # (b,c*2,f'+ws,h,w)
    return model_out


def write_kv_cache(model,z_predicted,model_kwargs,cls_free_guidance,noise):
    if cls_free_guidance == "text":
        z = torch.cat([z_predicted]*2,dim=0) # (2*b, c, ws, h, w)
    elif cls_free_guidance == "both":
        z = torch.cat([noise, z_predicted, z_predicted], dim=0) # (3b, c ,ws, h, w)
        assert False, '''
        TODO, for img_uncondition, t_input != t0, it should be t-level,
        but if so, the kv-cache can not be shared by all diffusion (denoising) steps
        '''
        mask_channel = torch.cat([zeros, ones, ones],dim=0)
        t_input = torch.cat([t_input, t_input0,t_input0],dim=3)
        model.write_kv_cache(z,mask_channel,t-input,start_id)
    else:
        z = z_predicted
    
    start_id = model_kwargs["start_id"]
    write_len = z.shape[2]
    print(f"write_kv_cache for [{start_id}:{start_id+write_len})")
    model.write_kv_cache(z, **model_kwargs) # z , y, y_mask, predict_start_id
    # always use mask_channel = all ones, and t_input = t0 for kv_cache computing

   
def forward_with_cfg2_kv_cache(
    model,
    z,              # (b,c,f'+ws,h,w)
    timestep,       # (b,)
    txt_guidance_scale,
    img_guidance_scale,
    **model_kwargs # y, mask, mask_channel
):
    '''
    clean_prefix = True and clean_prefix_set_t0=True in this func
    modifications:
        - add cls_free_guidance for txt & img
        - modify z & timestep for clean_prefix & clean_prefix_set_t0
    '''

    # two guidance mode: txt & txt+img
    cls_free_guidance = None
    if txt_guidance_scale > 1.0:
        cls_free_guidance = "text"
    if img_guidance_scale > 1.0:
        cls_free_guidance = "both"
        assert txt_guidance_scale > 1.0, "turn on txt guidance when using img guidance"
    
    n_frames_input = z.shape[2] # f' + window_size
    
    t_input = timestep
    if cls_free_guidance == "text":
        z = torch.cat([z]*2, dim=0)
        t_input = torch.cat([t_input]*2, dim=0)
    elif cls_free_guidance == "both":
        z = torch.cat([z]*3,dim=0)
        t_input = torch.cat([t_input]*2, dim=0)
        assert False, "TODO"
        # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
    
    model_out = model.forward_kv_cache(z, t_input, **model_kwargs) # (b,c*2,ws,h,w)

    # split prediction of mean and variance
    out_channel = model_out.shape[1] //2
    noise_pred, var_pred = model_out[:, :out_channel], model_out[:, out_channel:] # (b,c,ws,h,w)

    # perform guidance (only for mean), we do not perform guidance for variance 
    if cls_free_guidance == "text":
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2,dim=0)
        noise_pred = noise_pred_uncond + txt_guidance_scale * (noise_pred_text - noise_pred_uncond) # (b,c,ws,h,w)
        _,var_pred = var_pred.chunk(2,dim=0) # (b,c,ws,h,w)
        
    elif cls_free_guidance == "both":
        # bath[0]: img_uncond + txt_uncond; batch[1]: img_cond + txt_uncond; batch[2]: img_cond + txt_cond
        noise_pred_uncond, noise_pred_img, noise_pred_both = noise_pred.chunk(3,dim=0)
        noise_pred = noise_pred_uncond \
            + img_guidance_scale * (noise_pred_img - noise_pred_uncond) \
            + txt_guidance_scale * (noise_pred_both - noise_pred_img)
        _,_,var_pred = var_pred.chunk(3,dim=0)

    model_out = torch.cat([noise_pred,var_pred],dim=1) # (b,c*2,ws,h,w)
    return model_out



def sum_flat(tensor):
    return tensor.sum(
        dim=list(range(1,len(tensor.shape)))
    )