
val_data_cfg = dict(
    type="SkyTimelapseDatasetForEvalFVD",
    
    root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test",
    n_sample_frames = 33,
    image_size=(256,256),
    read_video = False,
    read_first_frame = True,
    class_balance_sample = True,
    num_samples_total = 2048,
)

batch_size = 4
num_workers = 8

val_scheduler  = dict(
    type="clean_prefix_iddpm",
    num_sampling_steps = 100,
    progressive_alpha = -1,
)

sample_cfgs = dict(
    width = 256,
    height = 256,
    num_frames = val_data_cfg["n_sample_frames"],
    auto_regre_chunk_len = 8,
    txt_guidance_scale = 1.0,
    img_guidance_scale = 1.0,
)


