
# pre-trained weight https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth




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
txt_dropout_prob = 0.1
img_dropout_prob = 0
progressive_alpha = -1
prefix_perturb_t = 100 # max diffusion step is 1000

prefix_min_len = 1
fix_ar_size = True
ar_size = 8

reweight_loss_const_len = None
reweight_loss_per_frame = False # These two will be disabled if `fix_ar_size=True`

############# exp dir 
_image_size = train_data_cfg["aspect_ratio_buckets"][0]
_exp_tag = "{}x{}x{}fi{}ArSize{}".format(
    train_data_cfg["n_sample_frames"],
    _image_size[0],_image_size[1],
    train_data_cfg["sample_interval"],
    ar_size
)
outputs = f"/data/CausalSTDiT_working_dir/depth14_{_exp_tag}_overfit_beach"

_CKPT_PixArt512x512= "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/PixArt-XL-2-512x512.pth"
_CKPT_OpenSORA16x512x512 = "/home/gkf/LargeModelWeightsFromHuggingFace/opensora/OpenSora-v1-HQ-16x512x512.pth"
_CKPT_SD_VAE_FT_EMA="/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/sd-vae-ft-ema"
_CKPT_T5_V_1_1_XXL = "/home/gkf/LargeModelWeightsFromHuggingFace/PixArt-alpha/t5-v1_1-xxl"

#### model configs
resume_from_ckpt = None
model = dict(
    type="CausalSTDiT2-Base",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=_CKPT_OpenSORA16x512x512, # https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth
    freeze = None,
    from_scratch = None,

    class_dropout_prob = txt_dropout_prob,
    temp_extra_in_channels = 1,
    temp_extra_in_all_block = False,
    temporal_max_len = 64,
    relative_tpe_mode = "cyclic",
    enable_flashattn = True,
    enable_layernorm_kernel = False,
    enable_sequence_parallelism = False,
    cross_frame_attn = "prev_prefix_1",
    t_win_size = 0,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained=_CKPT_SD_VAE_FT_EMA,
)
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

batch_size = 1
accumulation_steps = 4
lr = 2e-5
grad_clip = 1.0

epochs = 20
log_every_step = 5
ckpt_every_step = 20
validation_every_step = 20
validate_before_train = True

validation_configs = dict(
    scheduler = dict(
        type="clean_prefix_iddpm",
        num_sampling_steps = 100,
        progressive_alpha = progressive_alpha,
    ),
    enable_kv_cache = True,
    kv_cache_dequeue = True,
    kv_cache_max_seqlen = train_data_cfg["n_sample_frames"],
    sample_cfgs = dict(
        width = 256,
        height = 256,
        num_frames = 49,
        auto_regre_chunk_len = ar_size,
        txt_guidance_scale = 7.5,
        img_guidance_scale = 1.0,
        seed = "random",
    ),
    examples = [
        dict(
            prompt =  "a slow moving camera view of surfboard on the beach, with sea waves",
            first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach1.mp4.1st_frame.jpg",

            # the following configs will over-write those in `sample_cfgs`:
            txt_guidance_scale = 7.0,
            seed = 123,
        ),
        dict(
            prompt =  "Blue sky and white clouds, with waves hitting the beach",
            first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach2.mp4.1st_frame.jpg",
        )
    ],
    examples_json = "/path/to/val_examples.json", # if examples is None, we use examples from `examples_json`
)