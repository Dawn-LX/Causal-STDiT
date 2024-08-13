import torch
from einops import rearrange
import torch.nn as nn

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def demo():
    hidden_size = 1152
    eps = 1e-6

    norm2 = nn.LayerNorm(hidden_size, eps, elementwise_affine=False)

def all_close_1e_4(x,y):
    torch.allclose(x,y,atol=1e-4)

def show_x():
    # position_tag = "before_1stBlock"  # allclose
    # position_tag = "after_norm1"
    # position_tag = "after_norm1_scale_shift"
    # position_tag = "after_attn"
    position_tag = "after_attnCF"
    # position_tag = "before_norm2"



    x = torch.load(f"working_dirSampleOutput/_debug_KVcache/wo_kv_cache_{position_tag}_xBTSC.pt")
    x_clean = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_clean_xBTSC.pt")
    x_denoise = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_denoise_xBTSC.pt")
    print(x.shape,x_clean.shape,x_denoise.shape)
    print(x[0,:,0,100:105],x.shape)
    x2 = torch.cat([x_clean,x_denoise],dim=1)
    print(x2[0,:,0,100:105],x2.shape)

    num_eq = (x2 == x).sum()
    num_all = x2.numel()

    print(torch.allclose(x2,x))
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)

    print("-="*80)

    B,T,S,C = x.shape
    T_c = x_clean.shape[1]
    T_n = x_denoise.shape[1]
    assert T == T_c + T_n

def show_scale_shift():
    B,T,S,C = 1,9,256,1152
    T_c = 1
    T_n = 8
    # working_dirSampleOutput/_debug_KVcache/with_kv_cache_after_norm1_denoise_scale_B1C.pt
    # working_dirSampleOutput/_debug_KVcache/with_kv_cache_after_norm1_denoise_shift_B1C.pt
    position_tag = "after_norm1"

    shift = torch.load(f"working_dirSampleOutput/_debug_KVcache/wo_kv_cache_{position_tag}_shift_BNC.pt")
    scale = torch.load(f"working_dirSampleOutput/_debug_KVcache/wo_kv_cache_{position_tag}_scale_BNC.pt")
    shift_clean:torch.Tensor = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_clean_shift_B1C.pt")
    scale_clean = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_clean_scale_B1C.pt")
    shift_denoise = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_denoise_shift_B1C.pt")
    scale_denoise = torch.load(f"working_dirSampleOutput/_debug_KVcache/with_kv_cache_{position_tag}_denoise_scale_B1C.pt")

    shift = rearrange(shift, "B (T S) C -> B T S C",T=T, S=S)
    scale = rearrange(scale, "B (T S) C -> B T S C",T=T, S=S)
    print(shift.shape,scale.shape)
    assert shift.shape == scale.shape
    assert shift.shape[1] == T_c + T_n

    print("x-"*40 + " scale shift clean " + "x-"*40)

    shift_clean0 = shift[:,:T_c,:,:]  # B 1 S C
    scale_clean0 = scale[:,:T_c,:,:]  # B 1 S C
    assert all(torch.allclose(shift_clean0[:,:,s,:],shift_clean0[:,:,0,:]) for s in range(S))
    assert all(torch.allclose(scale_clean0[:,:,s,:],scale_clean0[:,:,0,:]) for s in range(S))
    shift_clean0 = shift_clean0[:,:,0,:] # B 1 C
    scale_clean0 = scale_clean0[:,:,0,:] # B 1 C
    assert shift_clean.shape == shift_clean0.shape and scale_clean.shape == scale_clean0.shape
    print(shift_clean0.shape,scale_clean0.shape)

    num_eq = (shift_clean0 == shift_clean).sum()
    num_all = shift_clean.numel()
    print(torch.allclose(shift_clean0,shift_clean))
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)

    num_eq = (scale_clean == scale_clean0).sum()
    num_all = scale_clean.numel()
    print(torch.allclose(scale_clean,scale_clean0))
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)


    print("x-"*40 + " scale shift denoise " + "x-"*40)

    shift_denoise0 = shift[:,T_c:,:,:]  # B T_n S C
    scale_denoise0 = scale[:,T_c:,:,:]  # B T_n S C
    assert all(torch.allclose(shift_denoise0[:,:,s,:],shift_denoise0[:,:,0,:]) for s in range(S))
    assert all(torch.allclose(scale_denoise0[:,:,s,:],scale_denoise0[:,:,0,:]) for s in range(S))
    shift_denoise0 = shift_denoise0[:,:,0,:] # B T_n C
    scale_denoise0 = scale_denoise0[:,:,0,:] # B T_n C
    assert all(torch.allclose(shift_denoise0[:,t,:],shift_denoise0[:,0,:]) for t in range(T_n))
    assert all(torch.allclose(scale_denoise0[:,t,:],scale_denoise0[:,0,:]) for t in range(T_n))
    shift_denoise0 = shift_denoise0[:,:1,:] # B 1 C
    scale_denoise0 = scale_denoise0[:,:1,:] # B 1 C

    print(shift_denoise0.shape,scale_denoise0.shape)
    print(shift_denoise.shape,scale_denoise.shape)

    for c in range(C):
        allclose_shift = torch.allclose(shift_denoise0[:,:,c],shift_denoise[:,:,c])
        allclose_scale = torch.allclose(scale_denoise0[:,:,c],scale_denoise[:,:,c])
        if not allclose_shift:
            print(c,"shift",shift_denoise0[:,:,c],shift_denoise[:,:,c])
        if not allclose_scale:
            print(c,"scale",scale_denoise0[:,:,c],scale_denoise[:,:,c])


def show_temb1():
    B,T,S,C = 1,9,256,1152
    T_c = 1
    T_n = 8

    print("x-"*40 + " timestep embedding " + "x-"*40)


    temb = torch.load("working_dirSampleOutput/_debug_KVcache/wo_kv_cache_temb_B_T_C6.pt")
    temb_clean = torch.load("working_dirSampleOutput/_debug_KVcache/with_kv_cache_clean_temb_B_C6.pt")
    temb_denoise = torch.load("working_dirSampleOutput/_debug_KVcache/with_kv_cache_denoise_temb_B_C6.pt")

    print(temb.shape,temb_clean.shape,temb_denoise.shape)
    assert T_c ==1
    temb_clean0 = temb[:,:T_c,:][:,0,:]
    temb_denoise0 = temb[:,T_c:,:]
    assert all(torch.allclose(temb_denoise0[:,t,:],temb_denoise0[:,0,:]) for t in range(T_n))
    temb_denoise0 = temb_denoise0[:,0,:]
    print(temb_clean0.shape,temb_denoise0.shape)

    print(torch.allclose(temb_clean,temb_clean0),(temb_clean!=temb_clean0).sum())
    # assert torch.allclose(temb_clean,temb_clean0)

    print(torch.allclose(temb_denoise,temb_denoise0))

    temb_denoise0 = temb_denoise0.reshape(B,6,-1)
    temb_denoise = temb_denoise.reshape(B, 6, -1)
    num_eq = (temb_denoise0 == temb_denoise).sum()
    num_all = temb_denoise.numel()
    print(torch.allclose(temb_denoise0,temb_denoise))
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)


    shift_msa0, scale_msa0, gate_msa0, shift_mlp0, scale_mlp0, gate_mlp0 = temb_denoise0.chunk(6,dim=1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb_denoise.chunk(6,dim=1)  # (B, 1, C)

    for c in range(C):
        allclose_shift = torch.allclose(shift_msa0[:,:,c],shift_msa[:,:,c])
        allclose_scale = torch.allclose(scale_msa0[:,:,c],scale_msa[:,:,c])
        if not allclose_shift:
            print(c,"shift",shift_msa0[:,:,c],shift_msa[:,:,c])
        if not allclose_scale:
            print(c,"scale",scale_msa0[:,:,c],scale_msa[:,:,c])

def show_temb2():
    B,T,S,C = 1,9,256,1152
    T_c = 1
    T_n = 8

    print("x-"*40 + " timestep embedding outside block " + "x-"*40)

    temb = torch.load("working_dirSampleOutput/_debug_KVcache/wo_kv_cache_temb_BTC.pt")
    temb_clean = torch.load("working_dirSampleOutput/_debug_KVcache/with_kv_cache_clean_temb_BC.pt")
    temb_denoise = torch.load("working_dirSampleOutput/_debug_KVcache/with_kv_cache_denoise_temb_BC.pt")

    print(temb.shape,temb_clean.shape,temb_denoise.shape)
    assert T_c ==1
    temb_clean0 = temb[:,:T_c,:][:,0,:]  # (B,C)
    temb_denoise0 = temb[:,T_c:,:]
    assert all(torch.allclose(temb_denoise0[:,t,:],temb_denoise0[:,0,:]) for t in range(T_n))
    temb_denoise0 = temb_denoise0[:,0,:]
    print(temb_clean0.shape,temb_denoise0.shape)


    print(torch.allclose(temb_clean,temb_clean0))
    for c in range(C):
        allclose = torch.allclose(temb_clean[:,c],temb_clean0[:,c])
        if not allclose:
            print(c,temb_clean[:,c],temb_clean0[:,c])
    
    print(torch.allclose(temb_denoise0,temb_denoise))
    for c in range(C):
        allclose = torch.allclose(temb_denoise0[:,c],temb_denoise[:,c])
        if not allclose:
            print(c,temb_denoise0[:,c],temb_denoise[:,c])

def show_tpe():
    _tag = "_debug_KVcache_wo_CfAttn"
    # _tag = "_debug_KVcache"
    
    tpe0_chunk_start_ids = [0, 0, 0, 0, 8,24,32,40,48,56,64,72,80,88,96,104,112,120,128]
    # tpe0_len =           [9,17,25,33,33,33,33,...]
    tpe0_len = [9,17,25] + [33] * (len(tpe0_chunk_start_ids)-3)
    tpe0_path = "working_dirSampleOutput/{}/wo_kv_cache_tpe_ChunkStartIdx{:03d}Len{:03d}_1TC.pt"

    tpe_clean_chunk_start_ids = [0,1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,129,137,145,153] # len == 21
    # tpe_clean_len =           [1,8,8,8, 8, 8, 8, 8, 8, 8, 8, ...]
    tpe_clean_len = [1] + [8]*(len(tpe_clean_chunk_start_ids)-1)
    tpe_clean_path = "working_dirSampleOutput/{}/with_kv_cache_tpe_clean_ChunkStartIdx{:03d}_1TC.pt"

    tpe_denoise_chunk_start_ids = [1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,129,137,145,153] # len == 20
    # tpe_denoise_len =           [8,8,8, 8, 8, 8, 8, 8, 8, 8, ...]
    tpe_denoise_len = [8]*len(tpe_denoise_chunk_start_ids)
    tpe_denoise_path = "working_dirSampleOutput/{}/with_kv_cache_tpe_denoise_ChunkStartIdx{:03d}_1TC.pt"

    '''MaxCondLen=25
    arstep=0: 
        w/o kv-cache: tpe: [0-8]
        w/ kv-cache: cond_tpe: [0] tpe_denoise: [1-8]
    arstep=1:
        w/o kv-cache: tpe: [0-16]
        w/ kv-cache: tpe_clean: [0][1-8] tpe_denoise: [9-16]
    arstep=2:
        w/o kv-cache: tpe: [0-24]
        w/ kv-cache: tpe_clean: [0][1-8][9-16] tpe_denoise: [17-24]
    arstep=3:
        w/o kv-cache: tpe: [0-32]  (reached max_cond_len)
        w/ kv-cache: tpe_clean: [0][1-8][9-16][17-24] tpe_denoise: [25-32]
    arstep=4,5,...: as above
    '''
    for ar_step in range(6):
        tpe0 = torch.load(
            tpe0_path.format(_tag,tpe0_chunk_start_ids[ar_step],tpe0_len[ar_step])
        )
        tpe_clean = torch.load(
            tpe_clean_path.format(_tag,tpe_clean_chunk_start_ids[ar_step])
        )
        prev_tpe_clean = []
        for prev_start_id in tpe_clean_chunk_start_ids[:ar_step]:  # if arstep==0, this is empty
            tpe_clean_i = torch.load(
                tpe_clean_path.format(_tag,prev_start_id)
            )
            prev_tpe_clean.append(tpe_clean_i)
        if len(prev_tpe_clean) > 0:
            prev_tpe_clean = torch.cat(prev_tpe_clean,dim=1)
            tpe_clean = torch.cat([prev_tpe_clean,tpe_clean],dim=1)

        if tpe_clean.shape[1] > 25:
            tpe_clean = tpe_clean[:,-25:,:]

        tpe_denoise = torch.load(
            tpe_denoise_path.format(_tag,tpe_denoise_chunk_start_ids[ar_step])
        )

        tpe1 = torch.cat([tpe_clean,tpe_denoise],dim=1)
        print(ar_step,tpe0.shape,tpe_clean.shape,tpe_denoise.shape,torch.allclose(tpe0,tpe1))

        num_eq = (tpe0 == tpe1).sum()
        num_all = tpe1.numel()
        print(" >>>",num_eq,num_all,num_eq/num_all,num_all-num_eq)


if __name__ == '__main__':
    show_x()
    # show_scale_shift()
    # show_temb1()
    # show_temb2()
    # show_tpe()