import torch
import torchvision
def num_allclose(x,y,eps=1e-6):
    return (torch.abs(x-y) < eps).sum().item()


def main():
    # _tag = "_debug_KVcache_wo_CfAttn"
    _tag = "_debug_KVcache"
    N_atsteps = 4
    chunks_with_kvcahe = []
    path_with_kvcache = "working_dirSampleOutput/{}/with_kv_cache_denoised_chunk_arstep{:02d}_BCTHW.pt"
    for i in range(N_atsteps):
        chunk = torch.load(path_with_kvcache.format(_tag,i)) # BCTHW
        if i==0: print(chunk.shape,chunk.dtype,chunk.device)
        assert chunk.shape[0] == 1
        chunk = chunk[0] # CTHW
        chunks_with_kvcahe.append(chunk)
    # chunks_with_kvcahe = torch.stack(chunks_with_kvcahe,dim=0)
    # print(chunks_with_kvcahe.shape) # (N,C,T,H,W)

    chunks_wo_kvcahe = []
    path_with_kvcache = "working_dirSampleOutput/{}/wo_kv_cache_denoised_chunk_arstep{:02d}_BCTHW.pt"
    for i in range(N_atsteps):
        chunk = torch.load(path_with_kvcache.format(_tag,i)) # BCTHW
        if i==0: print(chunk.shape,chunk.dtype,chunk.device)
        assert chunk.shape[0] == 1
        chunk = chunk[0] # CTHW
        chunks_wo_kvcahe.append(chunk)
    # chunks_wo_kvcahe = torch.stack(chunks_wo_kvcahe,dim=0)
    # print(chunks_wo_kvcahe.shape) # (N,C,T,H,W)

    frame_id = 1 # frame_id=0 is the given first frame, it is not loaded here
    for ar_step in range(N_atsteps):
        chunk0 = chunks_wo_kvcahe[ar_step]
        chunk1 = chunks_with_kvcahe[ar_step]  # (C,T,H,W)
        num_eq = num_allclose(chunk0,chunk1,eps=1e-3)
        num_all = chunk0.numel()
        print(f"ar_step={ar_step}",num_eq,num_all,num_eq/num_all)
        # for i in range(8):
        #     frame0 = chunk0[:,i,:,:]
        #     frame1 = chunk1[:,i,:,:]
        #     num_eq = num_allclose(frame0,frame1,eps=1e-3)
        #     num_all = frame0.numel()
        #     print(f"  >>> frame={frame_id:03d}",num_eq,num_all,num_eq/num_all)
        #     frame_id+=1

def main_rgb():
    
    # _tag = "_debug_KVcache_wo_CfAttn"
    _tag = "_debug_KVcache"
    absdiff_path = f"working_dirSampleOutput/{_tag}/idx0_seed555_with_wo_kvcache.mp4.absdiff.mp4"
    frames, _, info = torchvision.io.read_video(absdiff_path, pts_unit='sec',output_format = "THWC")
    fps = info['video_fps']
    T = frames.shape[0]
    frames = frames.to(torch.float32)
    for i in range(T):
        absdiff = frames[i,:,:,:]
        absdiff_sum = int(absdiff.sum().item())
        absdiff_mean = absdiff.mean().item()
        eps=3
        num_eq = ((absdiff - torch.zeros_like(absdiff)) < eps).sum().item()
        num_all = absdiff.numel()

        print(i,absdiff_sum,absdiff_mean,num_eq/num_all)

def main_rgb2(abs_diff_video_path):
    frames, _, info = torchvision.io.read_video(abs_diff_video_path, pts_unit='sec',output_format = "THWC")
    fps = info['video_fps']
    T = frames.shape[0]
    frames = frames.to(torch.float32)
    chunk2Tids = [[0]] + [list(range(1+i*8,1+i*8 + 8)) for i in range(T//8)]
    # [0][1-8][9-16][17-24],...,[153-160]
    print(chunk2Tids)
    for i,Tids in enumerate(chunk2Tids):
        absdiff = frames[Tids,:,:,:]
        absdiff_sum = int(absdiff.sum().item())
        absdiff_mean = absdiff.mean().item()
        eps=3
        num_eq = ((absdiff - torch.zeros_like(absdiff)) < eps).sum().item()
        num_all = absdiff.numel()

        print(i,absdiff_sum,absdiff_mean,num_eq/num_all)

if __name__ == "__main__":
    # main()
    main_rgb2(
        "/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalCyclic_idx0_seed555_with_wo_kvcache.mp4.absdiff.mp4"
    )
    
    '''
    w/o cf-attn, max_cond_len=25
    ar_step=0 30623 32768 0.934539794921875 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [8] denoise: [1-8]
    
    ar_step=1 28656 32768 0.87451171875     
        latent >> cond: [0][1-8] denoise [9-16]
        tpe    >> cond: [0][1-8] denoise [9-16]
    
    
    ar_step=2 28332 32768 0.8646240234375
        latent >> cond: [0][1-8][9-16] denoise [17-24]
        tpe    >> cond: [0][1-8][9-16] denoise [17-24]

    ar_step=3 27428 32768 0.8370361328125
        latent >> cond: [0][1-8][9-16][17-24] denoise [25-32]
        tpe    >> cond: [0][1-8][9-16][17-24] denoise [25-32]

    ar_step=4 542 32768 0.01654052734375
        latent >> cond: [8][9-16][17-24][25-32] denoise [33-40]
        tpe    >> cond: [8][9-16][17-24][25-32] denoise [25-32]

        tpe:
        cond: [0-24] denoise [25-32]

    ar_step=5 310 32768 0.00946044921875

    w/ cf-attn
    ar_step=0 32217 32768 0.983184814453125
        cond: [0] denoise: [1-8]
    
    ar_step=1 567 32768 0.017303466796875
        cond: [0-8] denoise [9-16]

    ar_step=2 170 32768 0.00518798828125
        cond: [0-16] denoise [17-24]

    ar_step=3 98 32768 0.00299072265625
    ar_step=4 64 32768 0.001953125
    ar_step=5 54 32768 0.00164794921875
    '''
