
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

# /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test
train_data_cfg = dict(
    type="SkyTimelapseDataset",

    root=f"{_SKT_TIMELAPSE_ROOT}/sky_train",
    n_sample_frames = 16,
    image_size=(256,256),
    unified_prompt = "a beautiful sky timelapse"
)


################ exp configs for causal-stdit
clean_prefix = True
clean_prefix_set_t0 = True
progressive_alpha = -1
prefix_perturb_t = 100 # max diffusion step is 1000

prefix_min_len = 8
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
    from_pretrained=_CKPT_OpenSORA16x512x512, # https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth
    freeze = None,
    from_scratch = None,
    is_causal = False,

    caption_channels = 0,
    temp_extra_in_channels = 0,
    temp_extra_in_all_block = False,
    max_tpe_len = 16,
    relative_tpe_mode = None,
    enable_flashattn = True,
    enable_layernorm_kernel = False,
    enable_sequence_parallelism = False,
    cross_frame_attn = None,
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

batch_size = 4
accumulation_steps = 2
lr = 2e-5
grad_clip = 1.0

epochs = 10
log_every_step = 5
ckpt_every_step = 1000
validation_every_step = 500
validate_before_train = True
not_save_optimizer = True

_VAL_DATA_ROOT= f"{_SKT_TIMELAPSE_ROOT}/sky_test"
_num_given_frames = prefix_min_len
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
    enable_kv_cache = False,
    max_condion_frames = prefix_min_len,
    examples = [
        dict(
            prompt =  train_data_cfg["unified_prompt"],
            first_image =  [f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_000000{_i}.jpg" for _i in range(46,46+_num_given_frames)],

            # the following configs will over-write those in `sample_cfgs`:
            # seed = 123,
        ),
    ],
    examples_json = "/path/to/val_examples.json", # if examples is None, we use examples from `examples_json`
)
