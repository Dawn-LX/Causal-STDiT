import torch
from rotary_embedding_torch import RotaryEmbedding
from flash_attn import flash_attn_func
from opensora.utils.rope_llama_src import precompute_freqs_cis,apply_rotary_emb_q_or_k


def run_attn(use_rope=True):
    B, S, H = 128, 32, 1152
    N, D = 16, 72
    nheads,head_dim = N,D
    T = 33

    device = torch.device("cuda:0")

    rope = RotaryEmbedding(D).to(device=device, dtype=torch.bfloat16)
    
    rotary_emb=rope.rotate_queries_or_keys

    
    generator = torch.Generator(device=device)
    generator.manual_seed(100)
    qkv = torch.randn(size=(3,B,T,nheads,head_dim),dtype=torch.float16,generator=generator,device=device)
    softmax_scale = head_dim**-0.5

    q,k,v = qkv.clone().unbind(0) # (B, seqlen, #head, #dim)

    if use_rope:
        q = q.permute(0,2,1,3) # (B, seqlen, #head, #dim) -> (B, #head, seqlen, #dim)
        k = k.permute(0,2,1,3)

        q = rotary_emb(q) # (B, #heads, seqlen, #dim)
        k = rotary_emb(k)

        q = q.permute(0,2,1,3) # cvt back to (B, seqlen, #head, #dim)
        k = k.permute(0,2,1,3)
    
    
    x = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
    )
    print(x.shape)


def run_attn_LlamaRoPE(use_rope=True):
    device = torch.device("cuda:0")
    B, S, H = 128, 32, 1152
    N, D = 16, 72
    nheads,head_dim = N,D
    softmax_scale = head_dim**-0.5
    max_tpe_len = 33
    T_c,T_n = 25,8

    freqs = precompute_freqs_cis(head_dim,max_tpe_len).to(device)

    qkv = tuple(
        torch.randn(size=(B,_len,nheads,head_dim),dtype=torch.float16,device=device)
        for _len in [T_n,T_c+T_n,T_c+T_n]
    )
    q,k,v = qkv # (B, seqlen, #head, #dim)
    if use_rope:
        print(q.shape,k.shape,freqs.shape)
        q_len,k_len = q.shape[1],k.shape[1]
        
        maxL = freqs.shape[0]
        freqs_k = freqs[0:k_len]
        k = apply_rotary_emb_q_or_k(k,freqs_k)
        
        for q_start in [1,9,17,25,33,41]:
            q_end = min(q_start+q_len,maxL)
            freqs_q = freqs[q_end-q_len:q_end]
            print(f"q range : [{q_end-q_len}:{q_end}], freqs_q:{freqs_q.shape}")
            q = apply_rotary_emb_q_or_k(q,freqs_q)
        
        q_start=0
        q_len=1
        q = q[:,:q_len,:,:]
        q_end = min(q_start+q_len,maxL)
        freqs_q = freqs[q_end-q_len:q_end]
        print(f"q range : [{q_end-q_len}:{q_end}], freqs_q:{freqs_q.shape}")
        q = apply_rotary_emb_q_or_k(q,freqs_q)

    
    x = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
    )
    print(x.shape)


if __name__ == "__main__":
    
    # run_attn(True)
    run_attn_LlamaRoPE(True)
    '''
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    export /data9T/gaokaifeng/project/CausalSTDiT/tests/test_attn_rope.py
    
    PYTHONPATH=$PYTHONPATH:"/data9T/gaokaifeng/project/CausalSTDiT" python tests/test_attn_rope.py


    '''
