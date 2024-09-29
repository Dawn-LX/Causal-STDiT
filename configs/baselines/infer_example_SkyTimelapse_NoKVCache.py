_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse"
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
    auto_regre_steps = 7,
    seed = 666
)

# '''set them in configs/baselines/exps_list.py
max_condion_frames = 25
enable_kv_cache = False
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = max_condion_frames
# '''
dtype = "fp16"
enable_flashattn = True
# cross_frame_attn= None
# training:
# max_seqlen=33, cond: [1,9,17,25]

# infer:

# "first_image": "/home/gkf/project/CausalSTDiT/assets/videos/beach1.mp4.1st_frame.jpg"
examples = [
    dict(
        prompt =  None,
        first_image =  f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_00000046.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        auto_regre_steps=20,
        seed = 555
    ), 

    dict(
        prompt =  None,
        first_image =  f"{_VAL_DATA_ROOT}/LiWpE-zW14I/LiWpE-zW14I_1/LiWpE-zW14I_frames_00000871.jpg",
        # the following configs will over-write those in `sample_cfgs`:
        auto_regre_steps=20,
        seed = 555
    ),    
    
   
]