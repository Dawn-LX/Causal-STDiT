
# pre-trained weight https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth

num_frames = 65
frame_interval = 3
image_size = (256,256)

dtype="bf16"
grad_checkpoint = True
plugin = "zero2"  # "zero2-seq" for seq parrallel
sp_size = 1


################ exp configs for causal-stdit
clean_prefix = True
clean_prefix_set_t0 = True
txt_dropout_prob = 0.1
img_dropout_prob = 0.1
progressive_alpha = 2.0
prefix_perturb_t = 100 # max diffusion step is 1000

prefix_min_len = 1
fix_ar_size = True
ar_size = 16

reweight_loss_const_len = None
reweight_loss_per_frame = False # These two will be disabled if `fix_ar_size=True`

############# exp dir 
_exp_tag = "{}x{}x{}fi{}ArSize{}".format(
    num_frames,image_size[0],image_size[1],frame_interval,ar_size
)
outputs = f"working_dir/{_exp_tag}_train_demo"

#### model configs
resume_from_ckpt = None
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="/path/to/ckpt/OpenSora-v1-HQ-16x256x256.pth", # https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth
    freeze = None,
    from_scratch = None,

    class_dropout_prob = txt_dropout_prob,
    temp_extra_in_channels = 1,
    temp_extra_in_all_blocks = False,
    temporal_max_len = 256,
    relative_tpe_mode = "cyclic",
    enable_flashattn = True,
    enable_layernorm_kernel = False,
    enable_sequence_parrallelism = False,
    cross_frame_attn = "prev_prefix_3",
    t_win_size = 0,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/path/to/ckpt/stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="/path/to/ckpt/DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=False, # This is for model parallelism
)
scheduler = dict(
    type="clean_prefix_iddpm",
    timestep_respacing="",
    progressive_alpha = progressive_alpha,
)

#### data configs:
train_data = dict(
    
)
num_workers = 4,
sampler_seed = 2142

batch_size = 2
accumulation_steps = 3
lr = 2e-5
grad_clip = 1.0

epochs = 1000
log_every_step = 1
ckpt_every_step = 1000
validation_every_step = 500
validate_before_train = True

validation_configs = dict(
    scheduler = dict(
        type="clean_prefix_iddpm",
        num_sampling_step = 100,
        progressive_alpha = progressive_alpha,
    ),
    enable_kv_cache = True,
    sample_cfgs = dict(
        width = 256,
        height = 256,
        num_frames = 65,
        auto_regre_chunk_len = 16,
        txt_guidance_scale = 7.5,
        img_guidance_scale = 1.0,
        seed = "random",
    ),
    examples = [
        dict(
            prompt =  "a tiger is walking through a snow-covered forest",
            first_image =  "/path/to/first_img.png",

            # the following configs will over-write those in `sample_cfgs`:
            txt_guidance_scale = 7.0,
            seed = 123,
        )
    ],
    examples_json = "/path/to/val_examples.json", # if examples is None, we use examples from `examples_json`
)