import os
_ROOT_CKPT_DIR = os.getenv("ROOT_CKPT_DIR","/home/gkf/LargeModelWeightsFromHuggingFace") # or /data9T/gaokaifeng/LargeModelWeightsFromHuggingFace
_ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #
# /data/SkyTimelapse or /data9T/gaokaifeng/datasets/SkyTimelapse

_CKPT_PixArt512x512= f"{_ROOT_CKPT_DIR}/PixArt-alpha/PixArt-XL-2-512x512.pth"
_CKPT_OpenSORA16x512x512 = f"{_ROOT_CKPT_DIR}/opensora/OpenSora-v1-HQ-16x512x512.pth"
_CKPT_SD_VAE_FT_EMA=f"{_ROOT_CKPT_DIR}/PixArt-alpha/sd-vae-ft-ema"
_CKPT_T5_V_1_1_XXL = f"{_ROOT_CKPT_DIR}/PixArt-alpha/t5-v1_1-xxl"

_SKT_TIMELAPSE_ROOT = f"{_ROOT_DATA_DIR}/SkyTimelapse/sky_timelapse"


#### data configs:
train_data_cfg = dict(
    type="VideoTextDatasetFromJson",

    video_paths="./assets/videos",
    anno_jsons="./assets/overfit_beach_video.jsonl",
    n_sample_frames=33,
    sample_interval=2,
    condition_ths=None,
    aspect_ratio_buckets=[(256,256)],
    sample_repeats=3000,
)


################ exp configs for causal-stdit
clean_prefix = True
clean_prefix_set_t0 = True
txt_dropout_prob = 0.1
img_dropout_prob = 0
progressive_alpha = -1
prefix_perturb_t = 50 # max diffusion step is 1000

prefix_min_len = 1
fix_ar_size = True
ar_size = 8

reweight_loss_const_len = None
reweight_loss_per_frame = False # These two will be disabled if `fix_ar_size=True`


#### model configs
resume_from_ckpt = None
model = dict(
    type="CausalSTDiT2-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=_CKPT_OpenSORA16x512x512,
    freeze = None,
    from_scratch = None,
    is_causal = True,

    caption_channels = 0,
    temp_extra_in_channels = 0,
    temp_extra_in_all_block = False,
    max_tpe_len = 33,
    relative_tpe_mode = "cyclic",
    enable_flashattn = True,
    enable_layernorm_kernel = False,
    enable_sequence_parallelism = False,
    cross_frame_attn = "prev_prefix_3",
)


vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained=_CKPT_SD_VAE_FT_EMA,
)
if model["caption_channels"]==0:
    text_encoder = None
else:
    text_encoder = dict(
        type="t5",
        from_pretrained=_CKPT_T5_V_1_1_XXL,
        model_max_length=120,
        shardformer=False, # This is for model parallelism
    )

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
accumulation_steps = 4
lr = 2e-5
grad_clip = 1.0

epochs = 20
log_every_step = 5
ckpt_every_step = 500
validation_every_step = 100
validate_before_train = True


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
    kv_cache_max_seqlen = 25,
    kv_cache_dequeue = True, 
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
    ],
)
