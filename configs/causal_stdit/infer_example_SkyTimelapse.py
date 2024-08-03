_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse/sky_timelapse"
_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"


val_scheduler  = dict(
    type="clean_prefix_iddpm",
    num_sampling_steps = 100,
    progressive_alpha = -1,
)

sample_cfgs = dict(
    width = 256,
    height = 256,
    num_frames = 49,
    auto_regre_chunk_len = 8,
    txt_guidance_scale = 1.0,
    img_guidance_scale = 1.0,
    seed = "random"
)

enable_kv_cache = False
kv_cache_dequeue = True
kv_cache_max_seqlen = 9

# training:
# max_seqlen=33, cond: [1,9,17,25]

# infer:

examples = [
    dict(
        prompt =  None,
        first_image =  f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_00000046.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        seed = 123,
        num_frames=129,
    ),
    
]