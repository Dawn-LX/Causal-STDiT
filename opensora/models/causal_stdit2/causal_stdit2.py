import random
import functools
from typing import Union,Optional,List,Dict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    SeqParallelMultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint
from opensora.utils.rope_llama_src import precompute_freqs_cis,apply_rotary_emb_q_or_k

from .attention import (
    AttentionWithContext,
    SeqParallelAttentionWithContext,
)

from opensora.utils.debug_utils import envs
@torch.no_grad()
def _init_conv2d_eye(conv):
    assert isinstance(conv,nn.Conv2d)
    # stride=1, kernel_size=(1,1), padding=0

    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    dtype,device = conv.weight.dtype,conv.weight.device
    ch_out,ch_in,_,_ = conv.weight.shape # e.g., (320,321,1,1)
    extra_in_channels = ch_in - ch_out
    eye_ = torch.eye(ch_out,dtype=dtype,device=device)
    conv.weight[:,:ch_out,:,:] = eye_[:,:,None,None]

@torch.no_grad()
def _init_linear_eye(layer):
    assert isinstance(layer,nn.Linear)

    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    dtype,device = layer.weight.dtype,layer.weight.device
    ch_out,ch_in = layer.weight.shape # e.g., (320,321,1,1)
    extra_in_channels = ch_in - ch_out
    eye_ = torch.eye(ch_out,dtype=dtype,device=device)
    layer.weight[:,:ch_out] = eye_



class RotaryEmbForCacheQueue(nn.Module):
    def __init__(self,dim_per_attn_head,max_length) -> None:
        super().__init__()

        freqs = precompute_freqs_cis(dim_per_attn_head,max_length)
        self.register_buffer("freqs",freqs,persistent=False)
        self.q_start = 0
    
    def set_attn_q_start(self,q_start):
        self.q_start = q_start

    def forward(self,q,k):
        '''
        this func is designed for RoPE w/ kv-cache and w/ kv-cache dequeue
        it will be called inside Attention's forward

        Args:
            q (torch.Tensor): (bsz,len_q,n_heads,head_dim)
            k (torch.Tensor): (bsz,len_k,n_heads,head_dim)
            q_start (int): 

        Returns:
            RoPE applied q, k
        '''

        q_len,k_len = q.shape[1],k.shape[1]
        maxL = self.freqs.shape[0]

        freqs_k = self.freqs[0:k_len]
        k = apply_rotary_emb_q_or_k(k,freqs_k)
        
        q_start = 0 if self.training else self.q_start
        if envs.DEBUG_ROPE:
            print(f"self.training={self.training}, q_start={q_start}")
        q_end = min(q_start+q_len,maxL)
        '''
        e.g., 
        for training:
            q_start = 0 = k_start, q_len <= max_seqlen_train == max_tpe_len

        for auto-regre infer w/ kv-cache
            q_start = 1, 9, 17, 25, ... for forward w/ kv-cache; q_len=8 (denoise chunk_len)
            or q_start=0 and q_len=1 for writing 1st frame to kv-cache
        '''
        freqs_q = self.freqs[q_end-q_len:q_end]
        q = apply_rotary_emb_q_or_k(q,freqs_q)
        
        return q,k


class TimestepEmbedderExpand(TimestepEmbedder):
    def forward(self,t, dtype):
        # t: (bsz,) or (bsz,f)

        if t.ndim == 1:
            # same as original `TimestepEmbedder`
            t_freq = self.timestep_embedding(t,self.frequency_embedding_size)
        else:
            assert t.ndim == 2
            b,f = t.shape
            t = rearrange(t, 'b f -> (b f)') # [t_{b1,f1},t_{b1,f2},t_{b1,f3},t_{b2,f1},t_{b2,f2},t_{b2,f3},...]
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
            t_freq = rearrange(t_freq, '(b f) c -> b f c', b=b,f=f)

        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        
        t_emb = self.mlp(t_freq) # (bsz,c) or (bsz, f, c)
        return t_emb

class T2IFinalLayerExpand(T2IFinalLayer):
    '''
    consider different timesteps for different frames
    '''
    def forward(self,x,t,num_temporal=None):
        '''
        x: (b,f*h*w,c)
        t: (b,c) or (b, f, c)
        '''
        if t.ndim == 2:
            # same as orginal `T2IFinalLayer`
            shift,scale = (self.scale_shift_table[None] + t[:,None]).chunk(2,dim=1)
            # (1,2,C) + (B,1,C) --> (B,2,C) --> .chunk --> (B, 1, C)
        else:
            B,N,C = x.shape
            assert t.ndim==3

            S = N // num_temporal
            t = t[:,:,None,:].repeat(1,1,S,1) # (b,f,s,c)
            t = rearrange(t,'b f s c -> b (f s) c',b=B,f=num_temporal) # (b, f*h*w,c) == (B,N,C)

            scale_shift_by_t = self.scale_shift_table[None,None,:,:] + t[:,:,None,:] # (B, N, 2, C) + (B,N,1,C) --> (B,N,2,C)
            scale_shift_by_t = scale_shift_by_t.chunk(2, dim=2) # (B,N, 1, C)
            shift,scale = (ss.squeeze(2) for ss in scale_shift_by_t) # (B, N ,C)
        
        x = t2i_modulate(self.norm_final(x),shift,scale)
        x = self.linear(x)
        return x


class CausalSTDiT2Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        input_size,
        patch_size,
        mlp_ratio=4.0,
        drop_path=0.0,
        temp_extra_in_channels = 0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        spatial_attn_enhance = None,
        is_causal: bool = True, # set it to False for ablation
        with_cross_attn = True,
        rope : Optional[RotaryEmbForCacheQueue] = None,
        _block_idx = -1, # for debug
    ):
        super().__init__()
        self._block_idx = _block_idx
        self.with_cross_attn = with_cross_attn
        self.is_causal = is_causal
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_sequence_parallelism = enable_sequence_parallelism
        # assert (cross_frame_attn in ["first_frame","last_prefix",None]) or cross_frame_attn.startswith("prev_frames_")
        assert spatial_attn_enhance in ["first_frame",None] or spatial_attn_enhance.startswith("prev_frames_")
        self.spatial_attn_enhance = spatial_attn_enhance
        if spatial_attn_enhance is not None:
            self.spatial_attn_ctx_len = 1 if spatial_attn_enhance=="first_frame" else int(spatial_attn_enhance.split('_')[-1]) # e.g., prev_frames_3
        
        self.input_size = input_size
        self.patch_size = patch_size


        if enable_sequence_parallelism:
            temp_attn_cls = SeqParallelAttentionWithContext
            cross_attn_cls = SeqParallelMultiHeadCrossAttention
        else:
            temp_attn_cls = AttentionWithContext
            cross_attn_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = AttentionWithContext( # this does not need seq_parrallel
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flashattn,
            is_causal=False,
        )
        if self.with_cross_attn:
            self.cross_attn = cross_attn_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # temporal attention
        self.attn_temp = temp_attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=self.enable_flashattn,
            is_causal = self.is_causal,
            rope = rope
        )

        if self.spatial_attn_enhance is not None:
            self.attn_cf = AttentionWithContext(
                hidden_size,
                num_heads = num_heads,
                qkv_bias=True,
                enable_flash_attn=self.enable_flashattn,
                is_causal = False
            )

        
        self.temp_extra_in_channels = temp_extra_in_channels
        if self.temp_extra_in_channels > 0:
            self.attn_temp_pre_merge = nn.Linear(
                hidden_size + temp_extra_in_channels, hidden_size, bias=True
            )

    def forward(self, x, y, t, mask=None, tpe=None, mask_channel=None):
        '''
        x: (b,f*h*w,c)
        t: diffusion timestep's emb: (b,c*6) or (b,f,c*6)
        tpe: temporal PosEmb
        mask_channel: (b,1,f,1,1): temporal mask channel (1 for prefix, 0 for noisy latent)
        '''
        B, N, C = x.shape
        H, W = [self.input_size[i] // self.patch_size[i] for i in [1,2]] # T: complete length of the entire sp_group
        S = H * W
        T = N // S
        assert N % S == 0

        if self._enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            assert T % sp_size == 0
            T = T // sp_size
        
        if self.training:
            pass
        else:
            # this is deprecated, use forward_kv_cache at inference time
            assert not self._enable_sequence_parallelism
        
        # print(t.shape,self.scale_shift_table.shape)
        if t.ndim == 2:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1) # (1, 6, C) + (B, 6, C) --> (B, 6, C)
            ).chunk(6, dim=1) # each one has shape (B, 1, C)
        else:
            # t: (b, f, c*6); if seq_parrallel, d_t is the local seqlen for each sp_rank
            assert t.ndim == 3

            t=t[:,:,None,:].repeat(1,1,S,1) # (b,f,s,6*c)
            t= rearrange(t, 'b f s c -> b (f s) c', b=B,f=T)
            t=t.reshape(B,N,6,C)

            scale_shift_by_t = self.scale_shift_table[None,None,:,:] + t # (1, 1, 6, C) + (B, N, 6, C) --> (B, N, 6, C)
            scale_shift_by_t = scale_shift_by_t.chunk(6, dim=2) # (B,N, 1, C)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (ss.squeeze(2) for ss in scale_shift_by_t) # (B, N ,C)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa) # (B, N, C) x (B, 1, C) -> (B, N, C)


        ######### spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)
        x = x + self.drop_path(gate_msa * x_s)

        ######### cross-frame attn spatial branch
        if self.spatial_attn_enhance is not None:
            assert not self._enable_sequence_parallelism, "TODO"
            sae_mode = self.spatial_attn_enhance
            x_s_ = rearrange(x, "B (T S) C -> B T S C",T=T, S= S)
            if sae_mode == "first_frame":
                x_1st = x_s_[range(B),0,:,:] # (B, S, C)
                spatial_cond = x_1st[:,None,:,:].repeat(1,T,1,1) # (B, T, S, C)
            
            elif sae_mode.startswith("prev_frames_"): # e.g., prev_frames_3
                assert (prev_L := int(sae_mode.split('_')[-1])) >=1
                prefix_len = mask_channel.sum(dim=[1,2,3,4]) # (b,1,f,1,1) --> (b,)
                prefix_len = prefix_len.type(torch.long)
                assert torch.all(prefix_len >= 1), f"prefix_len={prefix_len}"
                prev_prefix = []
                for i in range(prev_L):
                    _i = prefix_len - 1 - i
                    _i = torch.maximum(_i,torch.zeros_like(_i)) # element-wise maximum, in case some batch has short prefix_len
                    prev_prefix.append(x_s_[range(B), _i, :,:]) # (B, S, C)
                prev_prefix = torch.cat(prev_prefix,dim=1) # (B, prev_L*S, C)
                _x_s_repeat = x_s_.repeat(1,1,prev_L,1) # B T (prev_L S) C
                spatial_cond = torch.where(
                    mask_channel[:,0,:,:,:].expand_as(_x_s_repeat).type(torch.bool),
                    _x_s_repeat,
                    prev_prefix[:,None,:,:].repeat(1,T,1,1) # B T (prev_L S) C 
                )
            else:
                raise NotImplementedError(f"self.spatial_attn_enhance={sae_mode} is not implemented")
            
            spatial_cond = rearrange(spatial_cond,"B T S C -> (B T) S C", T =T) # (B T) (prev_L S) C

            x_s_ = rearrange(x_s_,"B T S C -> (B T) S C", T =T, S = S)
            x_s_ = self.attn_cf(x_s_, context = spatial_cond, is_ctx_as_kv=False,debug_info="attn_cf_with_self_repeat")
            x_s_ = rearrange(x_s_, "(B T) S C -> B (T S) C", T =T, S = S)

            x =  x + self.drop_path(gate_msa * x_s_)


        ######### temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)
        if tpe is not None:
            x_t = x_t + tpe
        
        
        attn_temp_kwargs=dict()
        if self.is_causal == "partial":
            cond_len = int(mask_channel[0,0,:,0,0].sum().item())
            attn_temp_kwargs.update({"cond_len":cond_len})

        inject_mask_channel = self.temp_extra_in_channels > 0 # TODO: maybe inject other input (channel-wise concat)
        if inject_mask_channel:
            b,_,f,_,_ = mask_channel.shape
            mask_channel = mask_channel.reshape(b,1,f,1).repeat(1, S, 1, 1) # (B, S, T, 1)
            mask_channel = rearrange(mask_channel, "B S T C -> (B S) T C")

            # mask_channel_1 = mask_channel.reshape(b,f,1).repeat_interleave(S, dim=0) # (B*S, T, 1) <-- this has the same results
            # mask_channel_2 = mask_channel.reshape(b,f,1).repeat(S, 1, 1) # (B*S, T, 1) <-- this is WRONG

            assert mask_channel.shape[:2] == x_t.shape[:2]

            x_t = torch.cat([x_t,mask_channel],dim=-1) # (B S) T C+1
            x_t = self.attn_temp_pre_merge(x_t) # (B S) T C


        x_t = self.attn_temp(x_t,**attn_temp_kwargs) # (B S) T C
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=T, S= S)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        if self.with_cross_attn:
            x = x + self.cross_attn(x, y, mask)
            # print("use cross attn")

        # mlp
        
        x_m_ = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.drop_path(gate_mlp * self.mlp(x_m_))

        return x

    
    def forward_kv_cache(self,x, y, t, mask=None,tpe=None, mask_channel=None, cached_kv=(None,None), return_kv=False,return_kv_only=False):
        '''
        x: (b,f*h*w,c)
        t: diffusion timestep's emb: (b,c*6) or (b,f,c*6)
        tpe: temporal PosEmb
        mask_channel: (b,1,f,1,1): temporal mask channel this should be all zeros
        '''
        assert self.is_causal
        assert not self.training

        B, N, C = x.shape
        H, W = [self.input_size[i] // self.patch_size[i] for i in [1,2]] # T: complete length of the entire sp_group
        S = H *  W
        T = N // S # window_size

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1) # (1, 6, C) + (B, 6, C) --> (B, 6, C)
        ).chunk(6, dim=1) # each one has shape (B, 1, C)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa) # (B, N, C) x (B, 1, C) -> (B, N, C)


        # =======================================================================
        # spatial branch
        # =======================================================================
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)
        x = x + self.drop_path(gate_msa * x_s)
        
        
        cached_kv_s,cached_kv_t = cached_kv
        cached_kv_s: Union[None, torch.Tensor]  # B T_p S C*2
        cached_kv_t: Union[None, torch.Tensor]  # B T_accu S C*2
            
        # =======================================================================
        # cross-frame attn spatial branch
        # =======================================================================
        spatial_kv = None
        if self.spatial_attn_enhance is not None:
            x_s = rearrange(x, "B (T S) C -> (B T) S C",T=T,S=S)
            is_clean_x = return_kv==True # or is_clean_x = t==0
            T_p = self.spatial_attn_ctx_len
            
            if is_clean_x: # for writing clean latents to kv-cache
                assert cached_kv_s is None, "spatial kv-cache does not rely on previous spatial kv-cache"

                _x_s_repeat = x_s.repeat(1,T_p,1) # (B T) (T_p S) C
                x_s,spatial_kv = self.attn_cf(x_s, context=_x_s_repeat, is_ctx_as_kv=False, return_kv = True)
                x_s:torch.Tensor        # (B T) S C
                spatial_kv:torch.Tensor # (B T) S C*2

                spatial_kv = rearrange(spatial_kv,"(B T) S C -> B T S C",T=T, S= S)
                if self.spatial_attn_enhance == "first_frame":
                    spatial_kv = spatial_kv[:,:1,:,:]  # B 1 S C*2
                    # NOTE here `:1` index the relative 1st frame of the current chunk
                    # , so `spatial_kv` will only be written once for the 1st call (the true 1st frame of the video)
                    # refer to `CausalSTDiT2.write_kv_cache`
                else:
                    if T < (T_p:=self.spatial_attn_ctx_len):
                        # e.g., T==1 for write 1st frame to cache
                        spatial_kv = spatial_kv.repeat_interleave(T_p//T+1,dim=1)[:,:T_p,:,:]
                    else:
                        spatial_kv = spatial_kv[:,-T_p:,:,:]  # B T_p S C*2

                x_s = rearrange(x_s,"(B T) S C -> B (T S) C",T=T, S= S)
                x = x + self.drop_path(gate_msa * x_s)
            
            else: # for denoise, conditioned on cached spatial-kv
                assert cached_kv_s is not None  # B T_p S C*2
                if isinstance(T,torch.Tensor): # why ?
                    T = int(T) 
                assert  cached_kv_s.shape[1] == T_p
                cached_kv_s = rearrange(cached_kv_s,"B T_p S C -> B (T_p S) C", T_p=T_p)
                cached_kv_s = cached_kv_s[:,None,:,:].repeat_interleave(T,dim=1) # B T (T_p S) C*2
                cached_kv_s = rearrange(cached_kv_s,"B T S C -> (B T) S C", T=T) # (B T) (T_p S) C*2

                x_s = self.attn_cf(x_s, context=cached_kv_s, is_ctx_as_kv=True, return_kv = False,debug_info="attn_cf_with_kv_cache")
                x_s = rearrange(x_s,"(B T) S C -> B (T S) C",T=T, S= S)
                x = x + self.drop_path(gate_msa * x_s)


        # =======================================================================
        # temporal branch
        # =======================================================================
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)
        if tpe is not None:
            x_t = x_t + tpe
        
        attn_temp_kwargs = dict()
        if self.is_causal == "partial":
            is_clean_x = return_kv
            attn_temp_kwargs.update({"is_clean_x":is_clean_x})
        

        inject_mask_channel = self.temp_extra_in_channels > 0 # TODO: maybe inject other input (channel-wise concat)
        if inject_mask_channel:
            b,_,f,_,_ = mask_channel.shape
            mask_channel = mask_channel.reshape(b,1,f,1).repeat(1, S, 1, 1) # (B, S, T, 1)
            mask_channel = rearrange(mask_channel, "B S T C -> (B S) T C")
            assert mask_channel.shape[:2] == x_t.shape[:2]

            x_t = torch.cat([x_t,mask_channel],dim=-1) # (B S) T C+1
            x_t = self.attn_temp_pre_merge(x_t) # (B S) T C  == (b*h*w, ws, c)
        
        if cached_kv_t is not None:
            T_accu = cached_kv_t.shape[1] # B T_accu S C*2
            cached_kv_t = rearrange(cached_kv_t,"B T S C -> (B S) T C", T=T_accu)

        x_t,temporal_kv = self.attn_temp(x_t,context=cached_kv_t,is_ctx_as_kv=True,return_kv = True, **attn_temp_kwargs)
        x_t = rearrange(x_t,"(B S) T C -> B (T S) C", T=T, S=S)
        temporal_kv = rearrange(temporal_kv,"(B S) T C -> B T S C", T=T, S=S)
        
        x = x + self.drop_path(gate_msa * x_t)
        
        if return_kv and return_kv_only:
            # for the last block (save the computation of cross-attn)
            return None,spatial_kv,temporal_kv

        ######### cross-attn
        if self.with_cross_attn:
            if self._enable_sequence_parallelism:
                # avoid using seq parallel
                x = x + MultiHeadCrossAttention.forward(self.cross_attn,x,y,mask)
            else:
                x = x + self.cross_attn(x, y, mask)
            # print("use cross attn")
        # mlp
        # x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        x_m_ = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.drop_path(gate_mlp * self.mlp(x_m_))

        if return_kv:
            return (
                x, 
                spatial_kv, # B T_p S C*2
                temporal_kv # B T S C*2
            )
        else:
            return x

@MODELS.register_module()
class CausalSTDiT2(nn.Module):
    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        temp_extra_in_channels = 0,
        temp_extra_in_all_block= False, # TODO ideally remove this, we always set False
        max_tpe_len = 64,  # max length of temporal position embedding (tpe), tpe idx that exceeds this will start cyclic shift
        temporal_max_len = None, # this deprecated, now we use max_tpe_len
        relative_tpe_mode = None,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        spatial_attn_enhance = None,
        cross_frame_attn:str = None,
        is_causal: bool = True,
    ):
        super().__init__()
        if (cross_frame_attn is not None) and (spatial_attn_enhance is None):
            # support old-version code
            spatial_attn_enhance = cross_frame_attn.replace("prev_prefix_","prev_frames_")
        if temporal_max_len is not None:
            # support for old-version code
            max_tpe_len = temporal_max_len

        self.is_causal = is_causal
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        # self.num_temporal = input_size[0] // patch_size[0]
        # NOTE in our setting, num_temporal is dynamic, we will compute `num_temporal` in each run

        self.num_spatial = (input_size[1] // patch_size[1]) * (input_size[2] // patch_size[2])

        self.num_heads = num_heads
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale

        assert relative_tpe_mode in ["offset","sample","cyclic","rope",None] # use cyclic for infinite auto-regre generation
        self.relative_tpe_mode  = relative_tpe_mode
        if relative_tpe_mode == "rope":
            self.rope = RotaryEmbForCacheQueue(self.hidden_size//self.num_heads,max_tpe_len)
            
        self.max_tpe_len = max_tpe_len

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())
        # add a <BOV> token for img( prefix) dropout (for video generation  w/o given first frame, and img_cls_free_guidance)
        bov_token = torch.randn(size=(in_channels,input_size[1],input_size[2])) / (in_channels ** 0.5)
        self.register_buffer("bov_token", bov_token, persistent=True) # TODO make it trainable ?

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedderExpand(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        if caption_channels > 0:
            self.y_embedder = CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=approx_gelu,
                token_num=model_max_length,
            )
        else:
            self.y_embedder = None

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                CausalSTDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    self.input_size,
                    self.patch_size,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    temp_extra_in_channels = temp_extra_in_channels if (i==0 or temp_extra_in_all_block) else 0,
                    spatial_attn_enhance=spatial_attn_enhance,
                    is_causal= is_causal,
                    with_cross_attn =  caption_channels > 0,
                    rope = self.rope if relative_tpe_mode == "rope" else None,
                    _block_idx = i
                )
                for i in range(self.depth)
            ]
        )
        self.temp_extra_in_channels = temp_extra_in_channels
        self.temp_extra_in_all_block = temp_extra_in_all_block
        
        self.spatial_attn_enhance = spatial_attn_enhance
        assert spatial_attn_enhance in ["first_frame",None] or spatial_attn_enhance.startswith("prev_frames_") # e.g., prev_frames_3


        self.final_layer = T2IFinalLayerExpand(hidden_size,np.prod(self.patch_size),self.out_channels)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_rank = dist.get_rank(get_sequence_parallel_group())
        else:
            self.sp_rank = None
        
        self.KV_CACHE_MAX_SEQLEN = 128
        self._kv_cache_registered = False
        self.kv_cache_dequeue = True

    def get_relative_tpe(self,chunk_len,chunk_start_idx=None,with_kv_cache=False):
        mode = self.relative_tpe_mode
        max_tpe_len = self.pos_embed_temporal.shape[1] # i.e., self.max_tpe_len
        assert chunk_len <= max_tpe_len, f"chunk_len={chunk_len},max_tpe_len={max_tpe_len}"
        '''
        (full-attn,causal-attn) fixed tpe w/o kv-cache
        (full-attn,causal-attn) cyclic tpe w/o kv-cache
        causal-attn fixed tpe w/ kv-cache
        causal-attn cyclic tpe w/ kv-cache

        for w/o kv-cache or self.training:
            chunk_len = cond_len + denoise_len
        
        for w/o kv-cache (inference)
            chunk_len = denoise_len, cond_len = len(cached_kv)
        '''

        if with_kv_cache and mode is None:
            assert chunk_start_idx + chunk_len <= max_tpe_len
            # refer to `tests/debug_autoregre_enumerate.py`
        
        if mode is None: # fixed (absolute) tpe
            if self.training:
                assert chunk_start_idx is None or chunk_start_idx == 0
                tpe = self.pos_embed_temporal[:,:chunk_len,:]
                # `chunk_len` can be < `max_tpe_len`
            else:
                if not with_kv_cache:
                    assert chunk_len <= max_tpe_len
                    tpe = self.pos_embed_temporal[:,:chunk_len,:]
                    # chunk_start_idx can be > 0 when condition_frame starts dequeue
                    # but for fixed tpe, we do not need `chunk_start_idx`
                else:
                    T_accu = chunk_start_idx + chunk_len
                    tpe = self.pos_embed_temporal[:,:T_accu,:]
                    # T_accu can be 1,9,17,...,49, 
                    # and max_tpe_len maybe, e.g., max_tpe_len=33
                    tpe = tpe[:,-chunk_len:,:]
                '''
                we can remove the ablve if-else for w/ & w/o kv-cache and use the following code:
                
                T_accu = chunk_start_idx + chunk_len # for w/o kv-cache, chunk_start_idx=0 by default
                tpe = self.pos_embed_temporal[:,:T_accu,:]
                tpe = tpe[:,-chunk_len:,:] # for w/o kv-cache, this line is useless

                but to make the code easy-reading, we use the ablve if-else
                '''

        elif mode == "cyclic":
            if self.training:
                assert chunk_start_idx is None or chunk_start_idx == 0
                if chunk_len < self.max_tpe_len:
                    tpe_start = 0
                    # NOTE cyclic shift only happens when generated seq > max_tpe_len
                    # so we do not train the model to fit cyclic shifted tpe for short frame seq
                else:
                    tpe_start = random.randint(0,max_tpe_len-1) 
                    # NOTE chunk_len 固定为8的时候，每次random start的id并不总是 0,8,24,32,(再下一个ar-step start_id=1)
                    # i.e., tpe_start_choices = [0, 8, 16, 24, 32, 1, 9, 17, 25,  ...]
                    # refer to `tests/debug_tpe_cyclic_shift.py`
            else:
                tpe_start = chunk_start_idx
            
            tpe_ids = [i % max_tpe_len for i in range(tpe_start,tpe_start+chunk_len)]
            # print(tpe_ids, max_tpe_len, self.pos_embed_temporal)
            tpe = self.pos_embed_temporal[:,tpe_ids,:]
            if envs.DEBUG_COND_LEN:
                print(f"chunk_len={chunk_len},random_tpe_start={tpe_start},tpe_ids={tpe_ids}")
        elif mode == "rope":
            if self.training:
                assert chunk_start_idx is None or chunk_start_idx == 0
            else:
                # chunk_start_idx can be 0, 1, 9, 17, 25,33,41,...
                self.rope.set_attn_q_start(chunk_start_idx)
            
            # we will apple RoPE inside the Attention's forward
            return None
        else:
            raise NotImplementedError(f"rel_tpe_mode={mode} is not implemented")

        assert tpe.shape[1] == chunk_len, f"tpe.shape={tpe.shape},chunk_len={chunk_len}"

        return tpe

        
 
    def _check_input_shape(self,x):
        b,c,t,h,w = x.shape
        assert tuple(self.input_size[1:]) == (h,w), f"x.shape=={x.shape}; input_size=={self.input_size}"

    def forward(self, x, timestep, y, mask=None, mask_channel=None, x_temporal_start=None):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
                NOTE if clean_prefix_set_t0 = True
                timestep.shape == (B,T), and the timestep of clean_prefix is set to 0
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]
            mask_channel: (B, 1, T, 1, 1) # extra mask (temporal-axis) channel, injected before temporal self-attn, TODO: rename this for injecting other information

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        if self.training:
            '''for training, e.g, input img size = 256x256
                x.shape == (B, C, T, H, W) == (3,4,16,32,32), input_size==(16,32,32); path_size == (1,2,2)
                num_patches = (16/1) * (32/2) * (32/2) == 4096
                num_temporal == 16/1 == 16
                num_spatial == (32/2)*(32/2) = 256
            '''
            if envs.DEBUG_COND_LEN:
                cond_mask = mask_channel[0,0,:,0,0]
                # print(f"x.shape={x.shape}, cond_len={int(cond_mask.sum().item())}, cond_mask={cond_mask}")
        else:
            ''' for auto-regressive inference 
                each auto-regre step has different `num_temporal`, e.g., T = 17
                x.shape == (B, C, T, H, W) == (3,4,17,32,32), input_size==(17,32,32); path_size == (1,2,2)
                num_patches = (17/1) * (32/2) * (32/2) = 4352
                num_temporal == 17/1 == 17
                num_spatial == (32/2)*(32/2) = 256
            '''
            
            assert self.patch_size[0] == 1, "TODO, consdier temporal patchify for auto-regre infer"
            if self._kv_cache_registered:
                # add this, so that we donot call `forward_kv_cache` outside the model
                # i.e., always call model.forward, so that we can keep use scheduler's sample func without modification
                return self.forward_kv_cache(x,timestep,y,mask=mask)

        device = self.x_embedder.proj.weight.device
        dtype = self.x_embedder.proj.weight.dtype
        self._check_input_shape(x)
        num_temporal = x.shape[2]

        x = x.to(dtype)
        timestep = timestep.to(dtype)
        if y is not None: y = y.to(dtype)
        if mask_channel is not None: mask_channel = mask_channel.to(dtype)


        # embedding
        x = self.x_embedder(x)  # [B, N, C]  # N = (16/1)*(32/2)*(32/2)
        x = rearrange(x, "B (T S) C -> B T S C", T=num_temporal, S=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")


        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C] or [B, T, C]
        t_mlp = self.t_block(t)  # [B, C*6]
        y,y_lens = self.process_text_embeddings_with_mask(y,mask)

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            sp_group =  get_sequence_parallel_group()
            sp_size = dist.get_world_size(sp_group)

            mask_channel = torch.chunk(mask_channel, sp_size, dim=2)[self.sp_rank].contiguous() # (bsz, 1, num_temporal//sp_size, 1,1)
            x = split_forward_gather_backward(x,sp_group, dim=1, grad_scale="down")

            if t_mlp.ndim == 3:
                t_mlp = split_forward_gather_backward(t_mlp, sp_group, dim=1, grad_scale="down") # (B, sub_T, C*6)


        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                if (not self.training) and (self.relative_tpe_mode is not None):
                    assert x_temporal_start is not None
                tpe = self.get_relative_tpe(
                    chunk_len= num_temporal,
                    chunk_start_idx = x_temporal_start,
                    with_kv_cache=False
                )
                
                if self.enable_sequence_parallelism:
                    assert False, "TODO"
                    tpe = torch.chunk(tpe, sp_size, dim=1)[self.sp_rank].contiguous()
            else:
                tpe = None

            x = auto_grad_checkpoint(block, x, y, t_mlp, y_lens, tpe, mask_channel)
            

        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, sp_group, dim=1, grad_scale="up")
            # x: (B, sub_T*S, C) --> (B, T*S, C); sub_T = T //sp_size

        # x.shape: [B, N, C] == (bsz, f*h*w, C)

        # final process
        x = self.final_layer(x, t,num_temporal=num_temporal)  # [B, N, C=T_p * H_p * W_p * C_out]
        input_size = (num_temporal, self.input_size[1], self.input_size[2])
        x = self.unpatchify(x,input_size)  # [B, C_out, T, H, W]
        

        x = x.to(torch.float32) # cast to float32 for better accuracy
        return x

    
    def empty_kv_cache(self):
        # empty kv-cache to save GPU memory
        
        if self._kv_cache_registered:
            del self.cache_kv
            del self.cache_indicator

            if self.spatial_attn_enhance is not None:
                del self.spatial_ctx_kv

                if self.spatial_attn_enhance =="first_frame":
                    self._1st_frame_kv_written = False

            self._kv_cache_registered = False
    
    def reset_kv_cache(self):
        if self._kv_cache_registered:
            self.cache_kv.zero_()
            self.cache_indicator.zero_()
            if self.spatial_attn_enhance is not None:
                self.spatial_ctx_kv.zero_()

    @torch.no_grad()
    def write_kv_cache(self,clean_x,y,mask,start_id): 
        # support old version code

        L_cache_accu = self.cache_indicator.sum().item()
        assert start_id == L_cache_accu

        self.write_latents_to_cache(clean_x,y,mask)

    
    def pre_allocate_kv_cache(self,bsz,max_seq_len=None,kv_cache_dequeue=True):
        # support old version code
        bsz2 = bsz*self.num_spatial

        self.register_kv_cache(bsz,max_seq_len=max_seq_len,kv_cache_dequeue=kv_cache_dequeue)

    def register_kv_cache(self,bsz,max_seq_len=None,kv_cache_dequeue=True):
        '''NOTE bsz should take account into cls_free_guidance'''
        if self._kv_cache_registered:
            self.reset_kv_cache()
            return

        device = self.pos_embed_temporal.device
        dtype = self.pos_embed_temporal.dtype

        B = bsz
        S = self.num_spatial
        C = self.hidden_size

        if max_seq_len is None:
            max_seq_len = self.KV_CACHE_MAX_SEQLEN
            kv_cache_dequeue = True
        
        cache_kv = torch.zeros(
            size=(self.depth, B, max_seq_len, S, C*2),
            device=device,dtype=dtype
        )
        cache_indicator = torch.zeros(size=(max_seq_len,),device=device,dtype=torch.long)

        self.register_buffer("cache_kv",cache_kv,persistent=False) # (depth, B, max_seqlen,S, C*2)
        self.register_buffer("cache_indicator",cache_indicator,persistent=False) # (max_seqlen,)
        if envs.DEBUG_KV_CACHE:
            cache_ids = torch.as_tensor(list(range(max_seq_len))).to(device)
            self.register_buffer("cache_ids",cache_ids,persistent=False)

        self._kv_cache_registered = True
        self.kv_cache_dequeue = kv_cache_dequeue
        print(f"kv cache pre allocated, self.cache_kv : {self.cache_kv.shape}")

        if (sae_mode := self.spatial_attn_enhance) is not None:
            n_ctx_frames = 1 if sae_mode =="first_frame" else int(sae_mode.split('_')[-1]) # e.g., prev_frames_3
            spatial_ctx_kv = torch.zeros(
                size=(self.depth, B, n_ctx_frames, S, C*2),
                device=device,dtype=dtype
            )
            self.register_buffer("spatial_ctx_kv",spatial_ctx_kv,persistent=False) 
            if sae_mode =="first_frame":
                self._1st_frame_kv_written = False
            print(f"spatial_attn_enhance context kv pre allocated, with shape: {self.spatial_ctx_kv.shape}")

    
    def _fetch_kv_cache(self):
        
        L_cache_accu = self.cache_indicator.sum().item()
        if self.spatial_attn_enhance == "first_frame":
            assert self._1st_frame_kv_written == (L_cache_accu > 0)
        if L_cache_accu == 0:
            return None,None
        
        if self.spatial_attn_enhance is not None:
            spatial_kv = self.spatial_ctx_kv
        else:
            spatial_kv = None


        if L_cache_accu > len(self.cache_indicator): # this happens if kv_cache_dequeue
            L_cache_accu = len(self.cache_indicator)
        
        temporal_kv = self.cache_kv[:,:,:L_cache_accu,:,:] # D B T_accu S C
        if envs.DEBUG_KV_CACHE: 
            fetched_cache_ids = self.cache_ids[:L_cache_accu]
            print(f"fetched_cache_ids = {fetched_cache_ids}, L_cache_accu={L_cache_accu}")
        
        return (
            spatial_kv, # B T_p S C*2
            temporal_kv # B T_accu S C*2
        )

    def _write_kv_cache(self,spatial_kv,temporal_kv):
        ''' to write:
        # spatial_kv  # D B T_p S C*2
        # temporal_kv # D B T S C*2      (D=self.depth)
        '''

        ## for spatial kv
        if spatial_kv is not None:
            B = spatial_kv.shape[1]
            if self.spatial_attn_enhance == "first_frame":
                if self._1st_frame_kv_written:
                    # NOTE only write once for SAE mode == "first_frame"
                    return
                else:
                    self.spatial_ctx_kv[:,:B,:,:,:] = spatial_kv
                    self._1st_frame_kv_written = True
            else:
                self.spatial_ctx_kv[:,:B,:,:,:] = spatial_kv
                # we use `:B` in case that the last batch from dataloader has a smaller batch_size
        

        ## for temporal kv
        _,B,len_to_write = temporal_kv.shape[:3] # D B T S C*2

        L_cache_accu = self.cache_indicator.sum().item() # cache_indicator can be [1,1,1,1,1,5], i.e., L_cache_accu can > len(cache_indicator)
        if L_cache_accu + len_to_write > len(self.cache_indicator):
            print(" >>> kv_cache_dequeue")

            if L_cache_accu < len(self.cache_indicator):
                # this will happen when len_to_write > 1
                n_dequeue = L_cache_accu + len_to_write - len(self.cache_indicator) # This is WRONG for a very large L_cache_accu (i.e., dequeue to mach)
            else:
                # for L_cache_accu == len(self.cache_indicator) 
                # or L_cache_accu > len(self.cache_indicator) # L_cache_accu can be very large
                n_dequeue = len_to_write
            
            self.cache_kv = torch.roll(self.cache_kv,-n_dequeue,dims=2)
            self.cache_kv[:,:B,-len_to_write:,:,:] = temporal_kv
            self.cache_indicator[-len_to_write:] += 1
            if envs.DEBUG_KV_CACHE:
                self.cache_ids = torch.roll(self.cache_ids,-n_dequeue,dims=2)
                cache_ids_to_write = self.cache_ids[-len_to_write:]
        
        else:
            self.cache_kv[:,:B,L_cache_accu:L_cache_accu+len_to_write,:,:] = temporal_kv
            # we use `:B` in case that the last batch from dataloader has a smaller batch_size
            self.cache_indicator[L_cache_accu:L_cache_accu+len_to_write] +=1
            if envs.DEBUG_KV_CACHE:
                cache_ids_to_write = self.cache_ids[L_cache_accu:L_cache_accu+len_to_write]
        
        if envs.DEBUG_KV_CACHE:
            print(f"cache_ids_to_write: {cache_ids_to_write}")
            print(f"after write_kv_cache, cache_indicator={self.cache_indicator}")
        
    
    def process_text_embeddings_with_mask(self,y,mask):
        if self.y_embedder is None:
            # here y can be None
            return None,None
        
        assert y is not None
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        B,_,N_token,C = y.shape
        if mask is not None:
            if mask.shape[0] != y.shape[0]: # this happens when using cls_free_guidance
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)

            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, C)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, C)  # (B, 1, N_token, C) --> (1, B*N_token, C)
        
        return y,y_lens

    @torch.no_grad()
    def write_latents_to_cache(self,clean_x,y,mask):
        '''only write kv cache once after finish the whole denoising loop (use clean_x)
        # clean_x.shape: (B, C, T, H, W)
        # build timestep embedding with all t0's embedding
        # build mask_channel with all ones
        '''
        
        device = self.x_embedder.proj.weight.device
        dtype = self.x_embedder.proj.weight.dtype
        
        x = clean_x.to(dtype)
        if y is not None: y = y.to(dtype)
        num_temporal = x.shape[2]

        mask_channel = torch.ones_like(x[:,:1,:,:1,:1]) # (B, 1, T, 1, 1)
        timestep = torch.zeros_like(x[:,0,0,0,0]) # (B,)

        # embedding
        x = self.x_embedder(x) # (B, N, C)
        x = rearrange(x, "B (T S) C -> B T S C",T=num_temporal,S = self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_mlp = self.t_block(t)  # [B, C*6]

        y,y_lens = self.process_text_embeddings_with_mask(y,mask)


        # blocks
        cached_kv_s,cached_kv_t = self._fetch_kv_cache() # this can be None for the 1st call (i.e., write the given 1st frame to kv-cache)
        # cached_kv_s,  # (depth, B, T_p, S, C*2)
        # cached_kv_t  # (depth, B, T_accu, S, C*2)
        cached_kv_s = None # overwtite it, spatial kv-cache does not rely on previous spatial-kv-cache
        
        kv_cache_to_write = []
        L_cache_accu = self.cache_indicator.sum().item() # this can be 0 for the 1st call (i.e., write the given 1st frame to kv-cache)
        for i, block in enumerate(self.blocks):
            block:CausalSTDiT2Block
            if i == 0:
                tpe = self.get_relative_tpe(
                    chunk_len=num_temporal,
                    chunk_start_idx=L_cache_accu,
                    with_kv_cache=True
                )
                mask_channel_input = mask_channel
            else:
                tpe = None
                mask_channel_input = mask_channel if self.temp_extra_in_all_block else None
            
            kv_s = None
            kv_t = None if cached_kv_t is None else cached_kv_t[i]
            cached_kv_i = (kv_s,kv_t)

            x, spatial_kv,temporal_kv = block.forward_kv_cache(
                x, y, t_mlp, y_lens, tpe, mask_channel_input, cached_kv=cached_kv_i,
                return_kv=True, return_kv_only = (i == len(self.blocks)-1)
            )

            kv_cache_to_write.append((
                spatial_kv,     # (B, T_p, S, C*2) 
                temporal_kv,    # (B, T, S, C*2) 
            ))
        
        if self.spatial_attn_enhance is not None:
            spatial_kv  = torch.stack([st_kv[0] for st_kv in  kv_cache_to_write],dim=0) # (depth, B, T_p, S, C*2)
        else:
            spatial_kv = None
        temporal_kv = torch.stack([st_kv[1] for st_kv in  kv_cache_to_write],dim=0) # (depth, B, T, S, C*2)
        self._write_kv_cache(spatial_kv,temporal_kv)
        
    
    @torch.no_grad()
    def forward_kv_cache(self,x,timestep,y,mask,start_id=None):
        assert not self.training
        # assert not self.enable_sequence_parallelism
        # NOTE `self.enable_sequence_parallelism` can be `True`, 
        # i.e., training with seq parallel, and `block.temp_attn` is `SeqParallelCausalSelfAttention`
        # but we don't use seq parallel at inference time,  
        # TODO: consider seq parallel

        device = self.x_embedder.proj.weight.device
        dtype = self.x_embedder.proj.weight.dtype
         
        num_temporal = x.shape[2]
        assert self.patch_size[0] == 1

        x = x.to(dtype)
        # timestep = timestep.to(dtype)
        if y is not None: y = y.to(dtype)
        mask_channel = torch.zeros_like(x[:,:1,:,:1,:1]) # (B, 1, T, 1, 1)


        # embedding
        x = self.x_embedder(x)  # [B, N, C]  # N = (16/1)*(32/2)*(32/2)
        x = rearrange(x, "B (T S) C -> B T S C", T=num_temporal, S=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_mlp = self.t_block(t)  # [B, C*6]

        y,y_lens = self.process_text_embeddings_with_mask(y,mask)


        # blocks
        cached_kv_s,cached_kv_t = self._fetch_kv_cache() # this can be None for the 1st call (i.e., write the given 1st frame to kv-cache)
        # cached_kv_s,  # (depth, B, T_p, S, C*2)
        # cached_kv_t  # (depth, B, T_accu, S, C*2)
        L_cache_accu = self.cache_indicator.sum().item() 
        assert L_cache_accu > 0 , "call `write_latents_to_cache` first"
        assert cached_kv_t is not None,  "call `write_latents_to_cache` first"

        if start_id is not None:
            # start_id is used in old-version code, remove this ideally
            assert start_id == L_cache_accu, f"start_id={start_id},L_cache_accu={L_cache_accu} " 
        
        for i, block in enumerate(self.blocks):
            block:CausalSTDiT2Block
            if i == 0:
                tpe = self.get_relative_tpe(
                    chunk_len=num_temporal,
                    chunk_start_idx=L_cache_accu,
                    with_kv_cache=True
                )

                mask_channel_input = mask_channel
            else:
                tpe = None
                mask_channel_input = mask_channel if self.temp_extra_in_all_block else None

            kv_s = None if cached_kv_s is None else cached_kv_s[i]  # it can be None when spatial_attn_enhance is None
            kv_t = cached_kv_t[i]
            cached_kv_i = (kv_s,kv_t)
            x = block.forward_kv_cache(x, y, t_mlp, y_lens, tpe, mask_channel_input,cached_kv=cached_kv_i, return_kv=False)

        
        # final process
        x = self.final_layer(x, t, num_temporal=num_temporal)  # [B, N, C=T_p * H_p * W_p * C_out]
        input_size = (num_temporal, self.input_size[1], self.input_size[2])
        x = self.unpatchify(x,input_size)  # [B, C_out, T, H, W]


        x = x.to(torch.float32) # cast to float32 for better accuracy
        return x
    
    def unpatchify(self, x, input_size):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [input_size[i] // self.patch_size[i] for i in range(3)]
        if not self.training:
            assert self.patch_size[0] == 1, "TODO: consider temporal patchify for auto-regression"
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x


    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        assert self.patch_size[0] ==1
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.max_tpe_len,
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

            if hasattr(block,"attn_temp_pre_merge"):
                _init_linear_eye(block.attn_temp_pre_merge)
            
            if hasattr(block,"attn_temp_window"):
                nn.init.constant_(block.attn_temp_window.proj.weight, 0)
                nn.init.constant_(block.attn_temp_window.proj.bias, 0)
            
            if hasattr(block,"attn_cf"):
                nn.init.constant_(block.attn_cf.proj.weight, 0)
                nn.init.constant_(block.attn_cf.proj.bias, 0)


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            if block.with_cross_attn:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module("CausalSTDiT2-XL/2")
def CausalSTDiT2_XL_2(from_pretrained=None,from_scratch=None, **kwargs):
    if "t_win_size" in kwargs: # support old version code
        kwargs.pop("t_win_size")
    model = CausalSTDiT2(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained,save_as_pt=True)
    if from_scratch is not None:
        assert from_scratch in ["temporal"] # TODO add other parts
        print(f"train {from_scratch} part from_scratch")
        if from_scratch == "temporal":
            print(f"  >>> re-initialize {from_scratch} part, discard the pre-trained {from_scratch} part")
            model.initialize_temporal()
    # print(model.pos_embed_temporal.shape)
    # assert False, f"model.pos_embed_temporal.shape={model.pos_embed_temporal.shape}"
    return model


# a tiny model for debug
@MODELS.register_module("CausalSTDiT2-Tiny") 
def CausalSTDiT2_Tiny(from_pretrained=None,from_scratch=None, **kwargs):
    model = CausalSTDiT2(depth=2, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    
    return model

# Base model
@MODELS.register_module("CausalSTDiT2-Base") 
def CausalSTDiT2_Base(from_pretrained=None,from_scratch=None, **kwargs):
    model = CausalSTDiT2(depth=14, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)

    return model