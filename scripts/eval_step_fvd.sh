if test -d "/data"; then
    # for server 10.130.129.11
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    
    export ROOT_CKPT_DIR="/home/gkf/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data"
    export SAMPLE_SAVE_DIR="/data/sample_outputs"
    echo "on server A100"
else
    # for server 10.130.129.34
    export CODE_ROOT="/data9T/gaokaifeng/project/CausalSTDiT"

    export ROOT_CKPT_DIR="/data9T/gaokaifeng/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data9T/gaokaifeng/datasets"
    export SAMPLE_SAVE_DIR="/data9T/gaokaifeng/video_gen_ddp_sample"
    echo "on server A6000"
fi

export PYTHONPATH=$PYTHONPATH:$CODE_ROOT
cd $CODE_ROOT
pwd


ABS_SAMPLE_CONFIG=${1}
EXP_DIR=${2}
export CUDA_VISIBLE_DEVICES=${3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export I3D_WEIGHTS_DIR="${CODE_ROOT}/_backup/common_metrics_on_video_quality-main/fvd"
# i3d_weights_path = f"{I3D_WEIGHTS_DIR}/styleganv/i3d_torchscript.pt" (for styleganv)
# i3d_weights_path = f"{I3D_WEIGHTS_DIR}/videogpt/i3d_pretrained_400.pt" (for videogpt)


python scripts/eval_fvd.py \
    --step_fvd \
    --sample_config $ABS_SAMPLE_CONFIG \
    --exp_dir $EXP_DIR \
    --batch_size 6 \
    --num_workers 4


<<comment


bash scripts/eval_fvd.sh \
/data/CausalSTDiT_working_dir/debug_inference/sampling_cfg_14831e3e05cfd0b1d0a97c2ff2a6b3f5_debug_inference.json \
working_dirSampleOutput/eval_fvd 3

## pure causal cyclic w/o PE

    # exp6
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_ddp_sample_17x256x256/sampling_cfg_8ee3f15f3b85f99b68acf4ae2a179419_exp6_ddp_sample_17x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6_17frames 0

    # exp6.2
    bash scripts/eval_step_fvd.sh \
    /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256/sampling_cfg_66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.2 1


    # exp6.4
    bash scripts/eval_step_fvd.sh \
    /data/CausalSTDiT_working_dir/exp6.4_ddp_sample_49x256x256/sampling_cfg_cb7782f3eddb9d360790070b6de9fe00_exp6.4_ddp_sample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.4 2

# full-attn fixed TPE

    # exp7.2
        bash scripts/eval_step_fvd.sh \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_ddp_sample_49x256x256/sampling_cfg_c392b906626f6a776cf81b2c39edf997_exp7.2_ddp_sample_49x256x256.json \
        working_dirSampleOutput/eval_step_fvd/exp7.2 0

        bash scripts/eval_step_fvd.sh \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_ddp_sample_49x256x256/sampling_cfg_c392b906626f6a776cf81b2c39edf997_exp7.2_ddp_sample_49x256x256.json \
        working_dirSampleOutput/eval_step_fvd/exp7.2_to_gt 0

    /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256/sampling_cfg_66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256.json
    "/data/sample_outputs/66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256",

comment