_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse"
_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"


val_data_cfg = dict(
    type="SkyTimelapseDatasetForEvalFVD",
    
    root=_VAL_DATA_ROOT,
    n_sample_frames = 16,
    image_size=(256,256),
    read_video = False,
    read_first_frame = True,
    class_balance_sample = True,
    num_samples_total = 2048,
)

batch_size = 4
num_workers = 8

# val_scheduler  = dict(
#     type="clean_prefix_iddpm",
#     num_sampling_steps = 100,
#     progressive_alpha = -1,
# )

# sample_cfgs = dict(
#     width = 256,
#     height = 256,
#     num_frames = 33, # TODO use auto-regression len
#     auto_regre_chunk_len = 8,
#     txt_guidance_scale = 1.0,
#     img_guidance_scale = 1.0,
# )

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
    auto_regre_steps = 2,
    seed = "random"
)

max_condion_frames = 25
enable_kv_cache = False
if enable_kv_cache:
    kv_cache_dequeue = True
    kv_cache_max_seqlen = max_condion_frames
# '''
dtype = "fp16"
enable_flashattn = True
