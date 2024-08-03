val_data_cfg = dict(
    type="SkyTimelapseDatasetForEvalFVD",
    
    root="/data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test",
    n_sample_frames = 16,
    image_size=(256,256),
    read_video = True,
    read_first_frame = False,
    class_balance_sample = True,
    num_samples_total = 2048,
)