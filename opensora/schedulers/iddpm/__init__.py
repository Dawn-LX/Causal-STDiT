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
            new_noise = (alpha / math.sqrt(1+alpha**2)) * prev_noise + (1/math.sqrt(1+alpha**2)) * noise[:,:,i:i+1,:,:]
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
    
    def sample_v2(
        self,
        model,
        z,
        prompts,
        device,
        model_kwargs = None,
        progress_bar = True,
        mask=None,
        **kwargs
    ):
        '''modifications:
        remove text_encoder here, prepare {y,mask} outside the sample func
        '''
        
        bsz = len(prompts)
        if self.progressive_alpha > 0:
            z = build_progressive_noise(self.progressive_alpha,z)
        
        forward = partial(forward_with_cfg_v2, model, cfg_scale=self.cfg_scale)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress_bar,
            device=device,
            mask=None, # not used, we use our own "cond_mask" in `model_kwargs`
        ) # (B, C, T, H, W)
        
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

def forward_with_cfg_v2(model, x, timestep, cfg_scale, **model_kwargs):
    '''
    modifications:
        - support both w/ cfg and w/o cfg
        - support w/o txt condition (i.e., y=None)
        - support condition frames (as prefix of x in T-axis)
    
    x: (B,C,T,H,W): noisy latent at t-level noise
    # T = T_c + T_n  when w/o kv_cache, i.e., x is the entire seq (partially noised) till current auto-regre step
    # T = T_n when w/ kv_cache, i.e., x is the noisy chunk to be denoised

    model_kwargs: {
        y,
        mask,

        # only for w/o kv-cache:
        mask_channel, # (B,1,T,1,1) ideally remove this, extra-channels for temp_attn, 1 for cond, 0 for noisy 
        x_cond,      #  (B,C,T_c,H,W) i.e., condition frames, manually assign them at each denoise timestep

        # for infer w/ kv-cache, 
        # the above args will be built and processed inside the model's kv-cache mechanism
    } 
    '''
    do_cfg = True if cfg_scale > 1.0 else False
    T = x.shape[2]
    
    if "x_cond" in model_kwargs:
        x_cond = model_kwargs.pop("x_cond") # (T,) the batch has a same condition_len
        T_c = x_cond.shape[2]
        x[:,:,:T_c,:,:] = x_cond
        t_input = timestep[:,None].repeat(1,T) # (B,T)
        t_input[:, :T_c] = 0
    else:
        # when kv-cache is enabled, we do not need x_cond since it's info is inside the cached kv
        t_input = timestep # (B,)

    if do_cfg:
        x = torch.cat([x]*2, dim=0)
        t_input = torch.cat([t_input]*2, dim=0)
    
    model_out = model.forward(x, t_input, **model_kwargs) # (B, C*2, T, H, W)

    # split prediction of mean and variance
    out_channel = model_out.shape[1] //2
    noise_pred, var_pred = model_out[:, :out_channel], model_out[:, out_channel:] # (B, C, T, H, W)

    # perform guidance (only for mean), we do not perform guidance for variance 
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2,dim=0)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond) # (B, C, T, H, W)
        _,var_pred = var_pred.chunk(2,dim=0) # (B, C, T, H, W)

    model_out = torch.cat([noise_pred,var_pred],dim=1) # (B, C*2, T, H, W)
    return model_out


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
    


def sum_flat(tensor):
    return tensor.sum(
        dim=list(range(1,len(tensor.shape)))
    )