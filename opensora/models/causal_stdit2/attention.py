import random
from typing import Union
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
try:
    from flash_attn import flash_attn_func
except:
    print("flash_attn is not installed")

from opensora.acceleration.communications import all_to_all, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group

from opensora.models.layers.blocks import LlamaRMSNorm
from opensora.utils.debug_utils import envs

def partial_causal_flash_attn(q,k,v, **kwargs):
    '''# NOTE spatial attn should not go here'''
    # q,k,v :   (B, N, num_heads,head_dim)
    seqlen_q = q.shape[1]
    seqlen_kv = k.shape[1]

    assert "causal" not in kwargs, "let me decide is causal or not inside `partial_causal_flash_attn`"

    use_partial = "cond_len" in kwargs
    use_dynamic = "is_clean_x" in kwargs
    assert use_partial ^ use_dynamic # XOR operation, there is one and only one `True`

    if use_partial: # partial causal, for training or infer w/o kv-cache
        assert seqlen_q == seqlen_kv # 
        T_c = kwargs.pop("cond_len")
        assert 0< T_c and T_c < seqlen_q
        # seqlen == T_c + T_n
        
        x1 = flash_attn_func(q,k,v,causal=True, **kwargs) # (B, seqlen, num_heads,head_dim)
        x2 = flash_attn_func(q,k,v,causal=False,**kwargs)
        x = torch.cat([x1[:,:T_c,:,:],x2[:,T_c:,:,:]],dim=1) # (B, seqlen, num_heads,head_dim)
        ''' or, use the following code to reduce seqlen of x2's full-attn
        q2 = q[:,T_c:,:,:]
        x2 = flash_attn_func(q2,k,v,causal=False,**kwargs) # (B, T_n, num_heads,head_dim)
        x = torch.cat([x1[:,:T_c,:,:],x2],dim=1)
        '''

    else: # dynamic causal or non-causal, for auto-regre inference w/ kv-cache
        
        # seqlen_q = T_n
        # seqlen_kv = T_c + T_n
        # NOTE here a special case is that writing 1st-frame to cache, where seqlen_q == seqlen_kv == 1 (or the `first_k` given frames)
        # so we do not `seqlen_q == seqlen_kv` to decide whether "partial" or "dynamic"

        if is_clean_x := kwargs.pop("is_clean_x"):
            causal = True # for writing kv-cache, i.e., after finish denoise the T_n part
        else:
            causal = False # for denoising chunk conditioned on clean prefix 
        x = flash_attn_func(q,k,v,causal=causal, **kwargs) # (B, T_n, num_heads,head_dim)
        ''' 
        NOTE for writing kv-cache, we should use causal-attention for the whole seqlen of T_c + T_n
        because now the chunk is clean, in the next auto-regre step, the denoise chunk is conditioned on the clean_x that
        was computed from causal attn. i.e., for clean_x, we always compute causal-attn
        i.e., we always compute causal-attn for the seqlen range of "clean" sequence part.
        '''

    return x

class AttentionWithContext(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        is_causal:  Union[bool,str] = False,  # True, False, or "partial"
        rope = None, # In our setting, we apply RoPE for temporal attention
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = is_causal
        self.rope = rope

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, is_ctx_as_kv = False, return_kv=False,**kwargs) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        # enable_flash_attn = self.enable_flash_attn and (N > B) # TODO
        qkv = self.qkv(x)
        if return_kv:
            # used for inference w/ kv-cache
            kv_before_norm = qkv[:,:,C:].clone()  # (B,N,2*C)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        if self.enable_flash_attn:
            qkv_permute_shape = (2,0,1,3,4) # (3,B,N,num_heads,head_dim)
        else:
            qkv_permute_shape = (2,0,3,1,4) # (3,B,num_heads,N,head_dim)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)

        if context is not None:

            N_c = context.shape[1]
            if is_ctx_as_kv:
                # (B S) T_c C*2     (for temporal), T_c can be max_kv_cache_len
                # (B T) (T_c S) C*2 (for spatial),  T_c is small, e.g., several previous frames

                kv_shape = (B, N_c, 2, self.num_heads, self.head_dim)
                kv = context.view(kv_shape).permute(qkv_permute_shape)
                extra_k,extra_v = kv.unbind(0)

                cat_dim = 1 if self.enable_flash_attn else 2
                k = torch.cat([extra_k,k],dim=cat_dim)
                v = torch.cat([extra_v,v],dim=cat_dim)
                '''# attn seqlen: 
                    for temporal_attn: N_c + N = T_c + T    
                    for spatial_attn:  N_c + N = T_c*S + S
                    NOTE the order matters for causal temporal_attn, we must let extra_k before k in `torch.cat`
                '''
                if kwargs.get("debug_info",None)=="attn_cf_with_kv_cache": # TODO ideally remove this
                    k = extra_k  # (B,N,num_heads,head_dim) for enable_flash_attn
                    v = extra_v
                    # print("<AttentionWithContext.forward>: use debug_info=attn_cf_with_kv_cache")

            else:
                qkv = self.qkv(context)
                qkv_shape = (B, N_c, 3, self.num_heads, self.head_dim)
                qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
                _,k,v = qkv.unbind(0) # overwrite k/v
                        
                        
        if self.rope is not None:
            # refer to RotaryEmbForCacheQueue
            # we apply RoPE after fetch kv-cache, i.e., we wrote kv-cache w/o RoPE
            q,k = self.rope(q,k)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flash_attn:
            if self.is_causal == "partial":
                x = partial_causal_flash_attn(
                    q,k,v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                    **kwargs
                )
            else:
                assert isinstance(self.is_causal,bool)
                x = flash_attn_func(
                    q,  # (B,N, num_heads, head_dim)
                    k,  # (B,N_cache + N, num_heads,head_dim)
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal = self.is_causal
                )
        else:
            dtype = q.dtype
            # k: (B,num_heads,N_cache + N,head_dim)
            q = q * self.scale # (B,num_heads,N,head_dim)
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            
            # assert not self.is_causal, "TODO: manually set a causal attn mask"
            if self.is_causal:
                if self.is_causal == "parital": assert False, '''
                    TODO, design the attn-mask for train & inference_with_kv_cache separately
                    refer to tests/test_causal_attn.py
                '''

                len_k = k.shape[2]
                len_q = q.shape[2]
                assert len_k >= len_q, "TODO:"
                attn_mask = torch.tril(torch.ones(size=(len_q,len_q)),diagonal=0).type(torch.bool) # 1 for keep, 0 for masked out
                if len_k > len_q:
                    attn_mask = torch.cat([
                        torch.ones(size=(len_q,len_k-len_q),dtype=torch.bool),
                        attn_mask
                    ],dim=1)
                
                attn_bias = torch.zeros(size=(len_q,len_k))
                attn_bias.masked_fill_(attn_mask.logical_not(),float("-inf"))
                attn_bias = attn_bias.to(device=attn.device,dtype=attn.dtype)

                attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_kv:
            return x,kv_before_norm
        else:
            return x


class SeqParallelAttentionWithContext(AttentionWithContext):

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, is_ctx_as_kv = False, return_kv=False) -> torch.Tensor:
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)

        B, SUB_N, C = x.shape  # for sequence parallel here, the SUB_N is a local sequence length
        N = SUB_N * sp_size
        qkv = self.qkv(x)
        if return_kv:
            kv_before_norm = qkv[:,:,C:].clone()
            assert False, "TODO: consider auto-regre for seq parallel"
        qkv_shape = (B, SUB_N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)

        

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flash_attn:
            qkv_permute_shape = (2,0,1,3,4) # (3,B,N,num_heads,head_dim)
        else:
            qkv_permute_shape = (2,0,3,1,4) # (3,B,num_heads,N,head_dim)

        qkv = qkv.permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)

        if context is not None:

            N_c = context.shape[1]
            if is_ctx_as_kv:
                # (B S) T_c C*2     (for temporal), T_c can be max_kv_cache_len
                # (B T) (T_c S) C*2 (for spatial),  T_c is small, e.g., several previous frames

                kv_shape = (B, N_c, 2, self.num_heads, self.head_dim)
                kv = context.view(kv_shape)
                kv = split_forward_gather_backward(kv,sp_group,dim=3, grad_scale="down") # [B*S T_c, 2, NUM_HEAD,  HEAD_DIM] -> [B*S T_c, 2, NUM_HEAD_PER_DEVICE, HEAD_DIM]
                kv = kv.permute(qkv_permute_shape)
                extra_k,extra_v = kv.unbind(0)

                cat_dim = 1 if self.enable_flash_attn else 2
                k = torch.cat([extra_k,k],dim=cat_dim)
                v = torch.cat([extra_v,v],dim=cat_dim)
                '''# attn seqlen: 
                    for temporal_attn: N_c + N = T_c + T    
                    for spatial_attn:  N_c + N = T_c*S + S
                    NOTE the order matters for causal temporal_attn, we must let extra_k before k in `torch.cat`
                '''
            else:
                qkv = self.qkv(context)
                qkv_shape = (B, N_c, 3, self.num_heads, self.head_dim)
                qkv = qkv.view(qkv_shape)
                qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)
                qkv = qkv.permute(qkv_permute_shape)
                _,k,v = qkv.unbind(0) # overwrite k/v
            
        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flash_attn:
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal = self.is_causal
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            
            assert not self.is_causal, "TODO add causal mask"

            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flash_attn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, SUB_N, C]
        x_output_shape = (B, SUB_N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_kv:
            return x, kv_before_norm
        else:
            return x

