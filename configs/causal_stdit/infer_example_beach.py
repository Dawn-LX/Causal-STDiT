
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
    auto_regre_steps = 7,
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
        prompt =  "a slow moving camera view of surfboard on the beach, with sea waves",
        first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach1.mp4.1st_frame.jpg",

        # the following configs will over-write those in `sample_cfgs`:
        auto_regre_steps = 7,
    ),
    dict(
        prompt =  "Blue sky and white clouds, with waves hitting the beach",
        first_image =  "/home/gkf/project/CausalSTDiT/assets/videos/beach2.mp4.1st_frame.jpg",
        auto_regre_steps = 7,
    ),
]
for _ar_steps in range(8,30):
    _tmp0 = examples[0].copy()
    _tmp1 = examples[1].copy()
    _tmp0.update(dict(auto_regre_steps=_ar_steps))
    _tmp1.update(dict(auto_regre_steps=_ar_steps))
    examples.extend([_tmp0,_tmp1])