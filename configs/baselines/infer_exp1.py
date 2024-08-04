_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse/sky_timelapse"
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
    auto_regre_steps = 3,
    seed = "random"
)

enable_kv_cache = False
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = 25
max_condion_frames = 25

# training:
# max_seqlen=33, cond: [1,9,17,25]

# infer:

examples = [
    dict(
        prompt =  None,
        first_image =  f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_00000046.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        auto_regre_steps=4,
    )
]

for _ar_steps in range(5,30):
    _tmp0 = examples[0].copy()
    _tmp0.update(dict(auto_regre_steps=_ar_steps))
    examples.extend([_tmp0])

'''# correct samples:
working_dirSampleOutput/exp1_full_attn_fix_tpe/val_samples/10000/idx0_seed423.mp4
'''