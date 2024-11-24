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
    auto_regre_steps = 10,
    seed = 111
)

# '''set them in configs/baselines/exps_list.py
max_condion_frames = 25
enable_kv_cache = True
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = max_condion_frames
# '''
dtype = "fp16"
enable_flashattn = True
# cross_frame_attn= None
prefix_perturb_t = 50

# training:
# max_seqlen=33, cond: [1,9,17,25]
examples = [
    dict(
        prompt =  "a slow moving camera view of surfboard on the beach, with sea waves",
        first_image =  "./assets/1st_frames/beach1.mp4.1st_frame.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        txt_guidance_scale = 7.0,
        seed = 123,
    ),
    dict(
        prompt =  "Blue sky and white clouds, with waves hitting the beach",
        first_image =  "./assets/1st_frames/beach2.mp4.1st_frame.jpg",
    ),
    dict(
        prompt =  "a camera view beach with weeds, slowly moving to the sea",
        first_image =  "./assets/1st_frames/beach3.mp4.1st_frame.jpg",
    )
]