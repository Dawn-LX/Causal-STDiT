import os

_ROOT_CKPT_DIR = os.getenv("ROOT_CKPT_DIR","/home/gkf/LargeModelWeightsFromHuggingFace") # or /data9T/gaokaifeng/LargeModelWeightsFromHuggingFace
_ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #
# /data/SkyTimelapse or /data9T/gaokaifeng/datasets/SkyTimelapse


_SKT_TIMELAPSE_ROOT = f"{_ROOT_DATA_DIR}/SkyTimelapse/sky_timelapse"
_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"


_CKPT_T5_V_1_1_XXL = f"{_ROOT_CKPT_DIR}/PixArt-alpha/t5-v1_1-xxl"


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
    auto_regre_steps = 7,
    seed = 666
)

text_encoder = dict(
    type="t5",
    from_pretrained=_CKPT_T5_V_1_1_XXL,
    model_max_length=120,
    shardformer=False, # This is for model parallelism
)

# '''set them in configs/baselines/exps_list.py
max_condion_frames = 8
enable_kv_cache = False
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = max_condion_frames
# '''
dtype = "fp16"

# update trained model keys: (use it cautiously for train/test mismatch)
enable_flashattn = False
# cross_frame_attn= None
caption_channels=4096
temp_extra_in_channels=1
num_given_frames = max_condion_frames
# training:
# max_seqlen=33, cond: [1,9,17,25]
examples = [
    dict(
        prompt =  "a beautiful sky timelapse",
        first_image =  f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_00000046.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        auto_regre_steps=20,
        seed = 555
    ), 
]