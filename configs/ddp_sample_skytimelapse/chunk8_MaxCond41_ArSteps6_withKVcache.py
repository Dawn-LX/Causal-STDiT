import os
_ROOT_CKPT_DIR = os.getenv("ROOT_CKPT_DIR","/home/gkf/LargeModelWeightsFromHuggingFace") # or /data9T/gaokaifeng/LargeModelWeightsFromHuggingFace
_ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #
# /data/SkyTimelapse or /data9T/gaokaifeng/datasets/SkyTimelapse

#### data configs:
_CKPT_PixArt512x512= f"{_ROOT_CKPT_DIR}/PixArt-alpha/PixArt-XL-2-512x512.pth"
_CKPT_OpenSORA16x512x512 = f"{_ROOT_CKPT_DIR}/opensora/OpenSora-v1-HQ-16x512x512.pth"
_CKPT_SD_VAE_FT_EMA=f"{_ROOT_CKPT_DIR}/PixArt-alpha/sd-vae-ft-ema"
_CKPT_T5_V_1_1_XXL = f"{_ROOT_CKPT_DIR}/PixArt-alpha/t5-v1_1-xxl"

_SKT_TIMELAPSE_ROOT = f"{_ROOT_DATA_DIR}/SkyTimelapse/sky_timelapse"
_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"

val_data_cfg = dict(
    type="SkyTimelapseDatasetForEvalFVD",
    
    root=_VAL_DATA_ROOT,
    n_sample_frames = 16,
    image_size=(256,256),
    read_video = False,
    read_first_frame = True,
    class_balance_sample = True,
    num_samples_total = 512,
)

batch_size = 4
num_workers = 8


scheduler  = dict(
    type="iddpm",
    num_sampling_steps = 100,
    cfg_scale = 1.0,
    progressive_alpha = -1,
)

sample_cfgs = dict(
    width = 256,
    height = 256,
    auto_regre_chunk_len = 8,
    auto_regre_steps = 6,
    seed = "random"
)

max_condion_frames = 41
enable_kv_cache = True
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = max_condion_frames
# '''
dtype = "fp16"
enable_flashattn = True
prefix_perturb_t = 50