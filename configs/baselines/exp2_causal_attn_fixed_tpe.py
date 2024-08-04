
#### data configs:

_CKPT_PixArt512x512= "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/PixArt-XL-2-512x512.pth"
_CKPT_OpenSORA16x512x512 = "/home/gkf/LargeModelWeightsFromHuggingFace/opensora/OpenSora-v1-HQ-16x512x512.pth"
_CKPT_SD_VAE_FT_EMA="/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/sd-vae-ft-ema"
_CKPT_T5_V_1_1_XXL = "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/t5-v1_1-xxl"

_SKT_TIMELAPSE_ROOT = "/data/SkyTimelapse/sky_timelapse/sky_timelapse"

# /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test
train_data_cfg = dict(
    type="SkyTimelapseDataset",

    root=f"{_SKT_TIMELAPSE_ROOT}/sky_train",
    n_sample_frames = 33,
    image_size=(256,256),
    unified_prompt = "a beautiful sky timelapse"
)


################ exp configs for causal-stdit
clean_prefix = True
clean_prefix_set_t0 = True
progressive_alpha = -1
prefix_perturb_t = 100 # max diffusion step is 1000

prefix_min_len = 1
ar_size = 8


############# exp dir 
_image_size = train_data_cfg["image_size"]
_exp_tag = "{}x{}x{}ArSize{}".format(
    train_data_cfg["n_sample_frames"],
    _image_size[0],_image_size[1],
    ar_size
)
outputs = f"/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_{_exp_tag}pp3"


#### model configs
resume_from_ckpt = None
model = dict(
    type="CausalSTDiT2-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=_CKPT_OpenSORA16x512x512, # https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth
    freeze = None,
    from_scratch = None,
    is_causal = True,

    caption_channels = 0,
    temp_extra_in_channels = 1,
    temp_extra_in_all_block = False,
    temporal_max_len = train_data_cfg["n_sample_frames"],
    relative_tpe_mode = None,
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

epochs = 20
log_every_step = 5
ckpt_every_step = 1000
validation_every_step = 200
validate_before_train = True


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
    enable_kv_cache = False,
    max_condion_frames = 25, # fixed tpe 的时候， max_condion_frames 必须是 model.temporal_max_len - auto_regre_chunk_len
    examples = [
        dict(
            prompt =  train_data_cfg["unified_prompt"],
            first_image =  f"{_VAL_DATA_ROOT}/07U1fSrk9oI/07U1fSrk9oI_1/07U1fSrk9oI_frames_00000046.jpg",

            # the following configs will over-write those in `sample_cfgs`:
            # seed = 123,
        ),
        dict(
            prompt =  train_data_cfg["unified_prompt"],
            first_image =  f"{_VAL_DATA_ROOT}/5WVkPmPTrv8/5WVkPmPTrv8_1/5WVkPmPTrv8_frames_00000061.jpg",
        )
    ],
    examples_json = "/path/to/val_examples.json", # if examples is None, we use examples from `examples_json`
)
