
#### data configs:

_CKPT_PixArt512x512= "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/PixArt-XL-2-512x512.pth"
_CKPT_OpenSORA16x512x512 = "/home/gkf/LargeModelWeightsFromHuggingFace/opensora/OpenSora-v1-HQ-16x512x512.pth"
_CKPT_SD_VAE_FT_EMA="/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/sd-vae-ft-ema"
_CKPT_T5_V_1_1_XXL = "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/t5-v1_1-xxl"

_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse/sky_timelapse"

# /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test
#### data configs:
train_data_cfg = dict(
    type="VideoTextDatasetFromJson",

    video_paths="/home/gkf/project/CausalSTDiT/assets/videos",
    anno_jsons="/home/gkf/project/CausalSTDiT/assets/overfit_beach_video.jsonl",
    n_sample_frames=25,
    sample_interval=3,
    condition_ths=None,
    aspect_ratio_buckets=[(256,256)],
    sample_repeats=3000,
)



################ exp configs for causal-stdit
clean_prefix = True
clean_prefix_set_t0 = True
progressive_alpha = -1
prefix_perturb_t = 100 # max diffusion step is 1000

prefix_min_len = 1
fix_ar_size = True
ar_size = 8

reweight_loss_const_len = None
reweight_loss_per_frame = False # These two will be disabled if `fix_ar_size=True`

############# exp dir 
_image_size = train_data_cfg["aspect_ratio_buckets"][0]
_exp_tag = "{}x{}x{}ArSize{}".format(
    train_data_cfg["n_sample_frames"],
    _image_size[0],_image_size[1],
    ar_size
)
outputs = f"/data/CausalSTDiT_working_dir/exp4_overfit_{_exp_tag}CfAttnPp3_tpe33"


#### model configs
resume_from_ckpt = None
model = dict(
    type="CausalSTDiT2-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=_CKPT_OpenSORA16x512x512, # https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth
    freeze = None,
    from_scratch = None,
    is_causal = "partial",

    caption_channels = 0,
    temp_extra_in_channels = 1,
    temp_extra_in_all_block = False,
    temporal_max_len = 33,
    relative_tpe_mode = "cyclic",
    enable_flashattn = True,
    enable_layernorm_kernel = False,
    enable_sequence_parallelism = False,
    cross_frame_attn = "prev_prefix_3",
    t_win_size = 0,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained=_CKPT_SD_VAE_FT_EMA,
)
text_encoder = None
scheduler = dict(
    type="clean_prefix_iddpm",
    timestep_respacing="",
    progressive_alpha = progressive_alpha,
)

dtype="bf16"
grad_checkpoint = True
plugin = "zero2"  # "zero2-seq" for seq parrallel
sp_size = 1

num_workers = 4
sampler_seed = 2142

batch_size = 2
accumulation_steps = 2
lr = 2e-5
grad_clip = 1.0

epochs = 10
log_every_step = 5
ckpt_every_step = 1000
validation_every_step = 200
validate_before_train = False
not_save_optimizer=True

_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"
validation_configs = dict(
    scheduler  = dict(
        type="iddpm",
        num_sampling_steps = 100,
        cfg_scale = 1.0,
        progressive_alpha = progressive_alpha,
    ),
    sample_cfgs = dict(
        width = 256,
        height = 256,
        auto_regre_chunk_len = ar_size,
        auto_regre_steps = 6,
        seed = "random"
    ),
    enable_kv_cache = True,
    kv_cache_max_seqlen = 33,
    kv_cache_dequeue = True, 
    examples = [
        dict(
            prompt =  None,
            first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach1.mp4.1st_frame.jpg",

            # the following configs will over-write those in `sample_cfgs`:
        ),
        dict(
            prompt =  None,
            first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach2.mp4.1st_frame.jpg",
        )
    ],
    examples_json = "/path/to/val_examples.json", # if examples is None, we use examples from `examples_json`
)
