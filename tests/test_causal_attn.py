import torch
torch.set_printoptions(linewidth=160)
from flash_attn import flash_attn_func



def causal_attn_demo(enable_flash_attn = True,is_causal=True):
    
    B,C,T,H,W = 2,128,8,1,1
    T_c = 5
    
    num_heads = 1
    head_dim = C 
    scale = head_dim**-0.5

    device=torch.device("cuda:0")
    generator = torch.Generator(device=device)
    generator.manual_seed(100)

    # x: (B,C,T,H,W) -- > (B H W) T C
    qkv = torch.randn(size=(3,B,T,head_dim),dtype=torch.float16,generator=generator,device=device)
    cache_kv = torch.randn(size=(2,B,T_c,head_dim),dtype=torch.float16,generator=generator,device=device)
    print(qkv[0,0,:,0])
    

    
    attn_mask = torch.tril(torch.ones(size=(T,T)),diagonal=0).type(torch.bool) # 1 for keep, 0 for masked out
    attn_mask = torch.cat([
        torch.ones(size=(T,T_c),dtype=torch.bool),
        attn_mask
    ],dim=1)
    print(attn_mask.float())

    attn_bias = torch.zeros(size=(T,T+T_c))
    attn_bias.masked_fill_(attn_mask.logical_not(),float("-inf"))

    # attn_bias = torch.triu(-1e4*torch.ones(size=(T,T)),diagonal=1)
    

    if enable_flash_attn:
        q,k,v = qkv.unbind(0)
        cache_k,cache_v = cache_kv.unbind(0)

        q = q[:,:,None,:] # (batch_size, seqlen, nheads, headdim); 
        k = k[:,:,None,:]
        v = v[:,:,None,:]
        cache_k = cache_k[:,:,None,:]
        cache_v = cache_v[:,:,None,:]

        k = torch.cat([cache_k,k],dim=1)
        v = torch.cat([cache_v,v],dim=1)
        print(q.shape,v.shape,v.shape)

        x = flash_attn_func(
            q, # (batch_size, seqlen, nheads, headdim)
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            causal = is_causal
        )
        x = x[:,:,0,:] #  (batch_size, seqlen, headdim)
    else:
        q,k,v = qkv.unbind(0)
        cache_k,cache_v = cache_kv.unbind(0)
        q = q[:,None,:,:] # (batch_size, nheads, seqlen, headdim); 
        v = v[:,None,:,:]
        k = k[:,None,:,:]
        cache_k = cache_k[:,None,:,:]
        cache_v = cache_v[:,None,:,:]
        k = torch.cat([cache_k,k],dim=2)
        v = torch.cat([cache_v,v],dim=2)

        print(q.shape,v.shape,v.shape)

        dtype = q.dtype
        q = q * scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        
        if is_causal:
            print(attn_bias)
            print(attn.shape,attn_bias.shape)
            attn = attn + attn_bias[None,None,:,:].cuda()


        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        # attn = self.attn_drop(attn)
        x = attn @ v

        x = x[:,0,:,:] #  (batch_size, seqlen, headdim)
        
    print(x[0,:,0],x.shape)

def causal_attn_demo2():
    device=torch.device("cuda:0")
    generator = torch.Generator(device=device)
    generator.manual_seed(100)

    B,head_dim = 2,144
    T = 9
    T_c = 5
    T_n = 4
    
    partial_causal_mask = torch.tril(torch.ones(size=(T,T)),diagonal=0).type(torch.bool) # 1 for keep, 0 for masked out
    partial_causal_mask[T_c:,:] = 1
    print(partial_causal_mask.float())

    attn_bias = torch.zeros(size=(T,T))
    # attn_bias.masked_fill_(partial_causal_mask.logical_not(),float("-inf"))
    attn_bias.masked_fill_(partial_causal_mask.logical_not(),-1e6)
    print(attn_bias)

    scale = 1 / (head_dim**0.5)
    qkv = torch.randn(size=(3,B,T,head_dim),dtype=torch.float16,generator=generator,device=device)
    
    print(qkv[0,0,:,0])
    
    # enable_flash_attn
    q,k,v = qkv.clone().unbind(0)
    q = q[:,:,None,:] # (batch_size, seqlen, nheads, headdim); 
    k = k[:,:,None,:]
    v = v[:,:,None,:]
    x1 = flash_attn_func(q,k,v,softmax_scale=scale,causal=True)[:,:,0,:] # (batch_size, seqlen, headdim); 
    x2 = flash_attn_func(q,k,v,softmax_scale=scale,causal=False)[:,:,0,:]
    x = torch.cat([x1[:,:T_c,:],x2[:,T_c:,:]],dim=1) # (batch_size, seqlen, headdim); 

    print(x[0,:,0],x.shape)
    x_flash_attn = x.clone()
    print("-="*80)

    # w/o flash_attn
    q,k,v = qkv.clone().unbind(0)
    q = q[:,None,:,:] # (batch_size, nheads, seqlen, headdim); 
    v = v[:,None,:,:]
    k = k[:,None,:,:]
    dtype = q.dtype
    q = q * scale
    attn = q @ k.transpose(-2, -1)  # translate attn to float32
    attn = attn.to(torch.float32)
    attn = attn + attn_bias.to(attn.device)
    attn = attn.softmax(dim=-1)
    attn = attn.to(dtype)  # cast back attn to original dtype
    x = attn @ v
    x = x[:,0,:,:] #  (batch_size, seqlen, headdim)

    print(x[0,:,0],x.shape)
    x_orig_attn = x.clone()
    print("-="*80)
    neq_ratio = (x_flash_attn != x_orig_attn).sum().item() / x_flash_attn.numel()
    print(neq_ratio, (x_flash_attn-x_orig_attn).max())
    # assert torch.allclose(x_flash_attn,x_orig_attn)
    for i in range(T):
        x1 = x_orig_attn[:,i,:]
        x2 = x_flash_attn[:,i,:]
        neq_ratio = (x1 != x2).sum().item() / x2.numel()
        print(i, neq_ratio, torch.abs(x2-x1).max())


def causal_attn_demo3():
    device=torch.device("cuda:0")
    generator = torch.Generator(device=device)
    generator.manual_seed(100)

    B,head_dim = 2,144
    T = 9
    T_c = 5
    T_n = 4

    scale = 1 / (head_dim**0.5)
    qkv = torch.randn(size=(3,B,T,head_dim),dtype=torch.float16,generator=generator,device=device)
    
    print(qkv[0,0,:,0])
    
    # enable_flash_attn
    q,k,v = qkv.clone().unbind(0)
    q = q[:,:,None,:] # (batch_size, seqlen, nheads, headdim); 
    k = k[:,:,None,:]
    v = v[:,:,None,:]
    x1 = flash_attn_func(q,k,v,softmax_scale=scale,causal=True)[:,:,0,:] # (batch_size, seqlen, headdim); 
    x2 = flash_attn_func(q,k,v,softmax_scale=scale,causal=False)[:,:,0,:]
    x = torch.cat([x1[:,:T_c,:],x2[:,T_c:,:]],dim=1) # (batch_size, seqlen, headdim); 

    print(x[0,:,0],x.shape)
    x_version1 = x.clone()
    print("-="*80)


    q,k,v = qkv.clone().unbind(0)
    q = q[:,:,None,:] # (batch_size, seqlen, nheads, headdim); 
    k = k[:,:,None,:]
    v = v[:,:,None,:]
    x1 = flash_attn_func(q,k,v,softmax_scale=scale,causal=True)[:,:,0,:] # (batch_size, seqlen, headdim); 
    
    q2 = q[:,T_c:,:,:] # (batch_size, T_c, nheads, headdim); 
    x2 = flash_attn_func(q2,k,v,softmax_scale=scale,causal=False)[:,:,0,:] # (batch_size, T_c, headdim); 
    print(x1.shape,x2.shape)
    x = torch.cat([x1[:,:T_c,:],x2],dim=1) # (batch_size, seqlen, headdim); 
    print(x[0,:,0],x.shape)
    x_version2 = x.clone()
    print("-="*80)
    assert torch.allclose(x_version1,x_version2)




def causal_attn_demo2_time():
    from tqdm import tqdm


    device=torch.device("cuda:0")
    

    B,head_dim = 2,144
    nheads = 8
    N_repeat = 100000
    scale = 1 / (head_dim**0.5)
    
    T = 128
    T_c = 96
    T_n = T - T_c
    
    partial_causal_bias = torch.triu(-10000*torch.ones(size=(T,T)),diagonal=1).to(device)
    partial_causal_bias[T_c:,:] = 0
    print(partial_causal_bias)

    
    generator = torch.Generator(device=device)
    generator.manual_seed(100)
    qkv = torch.randn(size=(3,B,T,nheads,head_dim),dtype=torch.float16,generator=generator,device=device)
    
    print(qkv[0,0,:,0])
    
    # enable_flash_attn
    for _ in tqdm(range(N_repeat)):
        q,k,v = qkv.clone().unbind(0)
        x1 = flash_attn_func(q,k,v,softmax_scale=scale,causal=True) # (batch_size, seqlen, nheads, headdim); 
        x2 = flash_attn_func(q,k,v,softmax_scale=scale,causal=False)
        x = torch.cat([x1[:,:T_c,:,:],x2[:,T_c:,:,:]],dim=1)

    print(x[0,:,0,0],x.shape)
    print("-="*80)

    # enable_flash_attn version-2
    for _ in tqdm(range(N_repeat)):
        q,k,v = qkv.clone().unbind(0)
        x1 = flash_attn_func(q,k,v,softmax_scale=scale,causal=True) # (batch_size, seqlen, nheads, headdim); 
        q2 = q[:,T_c:,:,:]
        x2 = flash_attn_func(q2,k,v,softmax_scale=scale,causal=False)
        x = torch.cat([x1[:,:T_c,:,:],x2],dim=1)

    print(x[0,:,0,0],x.shape)
    print("-="*80)

    

    # w/o flash_attn
    
    for _ in tqdm(range(N_repeat)):
        q,k,v = qkv.clone().unbind(0) # # (batch_size, seqlen, nheads, headdim); 
        q = q.permute(0,2,1,3) # # (batch_size, nheads, seqlen, headdim); 
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        
        dtype = q.dtype
        q = q * scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn + partial_causal_bias
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        x = attn @ v #  (batch_size,nheads, seqlen, headdim)

    print(x[0,0,:,0],x.shape)
    print("-="*80)


def attn_demo3():
    attn_mat1 = torch.randn(3,5)
    attn_mat2 = torch.randn(2,5)

    v = torch.randn(5,16)

    v1 = torch.cat([attn_mat1,attn_mat2],dim=0) @ v
    v2 = torch.cat([attn_mat1 @ v, attn_mat2 @ v],dim=0)

    print(v1)
    print(v2)
    assert torch.allclose(v1,v2)

if __name__ == "__main__":
    
    # print("-="*80)
    # causal_attn_demo(enable_flash_attn=False,is_causal=True)
    # print("-="*80)
    # causal_attn_demo(enable_flash_attn=True,is_causal=True)
    # attn_demo3()
    causal_attn_demo3()
    # causal_attn_demo2()
    causal_attn_demo2_time()
    # causal_attn_demo(enable_flash_attn=False,is_causal=False)
    # T = 6
    # diag = torch.triu(-10000*torch.ones(size=(T,T)),diagonal=1)
    # print(diag)
    # attn_demo3()
