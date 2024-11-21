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
    /data/CausalSTDiT_working_dir/exp6CfAttnpp3_ddp_sample_49x256x256/sampling_cfg_10b9d84d538171360734ae7130b697b1_exp6CfAttnpp3_ddp_sample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6_to_gt 3

    # exp6.2
    bash scripts/eval_step_fvd.sh \
    /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256/sampling_cfg_66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.2_to_gt 1

    # exp6.2 31k
    bash scripts/eval_step_fvd.sh \
    /data/CausalSTDiT_working_dir/exp6.2_31k_ddpsample49x256x256/sampling_cfg_965a66e6d6cc73936821500fb412156f_exp6.2_31k_ddpsample49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.2_31k_to_gt 1


    # exp6.4
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4ckpt11k_ddpsample_49x256x256/sampling_cfg_98b18250db41725f7ee7d0496483eed4_exp6.4ckpt11k_ddpsample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.4_11k 0

    # exp6.5
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.5ckpt11k_ddpsample_49x256x256/sampling_cfg_b4708ba8e8310df254f5a23c4d272c44_exp6.5ckpt11k_ddpsample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.5_11k_to_GT 2

    # exp6.6
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.6ckpt11k_ddpsample_49x256x256/sampling_cfg_a3c3670b452141906a9a07e2625f8323_exp6.6ckpt11k_ddpsample_49x256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp6.6_11k_to_gt 0

# full-attn fixed TPE

    # exp7.2
        bash scripts/eval_step_fvd.sh \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_ddp_sample_49x256x256/sampling_cfg_c392b906626f6a776cf81b2c39edf997_exp7.2_ddp_sample_49x256x256.json \
        working_dirSampleOutput/eval_step_fvd/exp7.2 0

        bash scripts/eval_step_fvd.sh \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_ddp_sample_49x256x256/sampling_cfg_c392b906626f6a776cf81b2c39edf997_exp7.2_ddp_sample_49x256x256.json \
        working_dirSampleOutput/eval_step_fvd/exp7.2_to_gt 0

    # exp7.4
    
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_ddp_sample_6stepx256x256/sampling_cfg_91d28521d49cdc4f885bbebb919b3975_exp7.4_ddp_sample_6stepx256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp7.4_to_gt 2
    
    # exp7.5
    bash scripts/eval_step_fvd.sh \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.5_ddp_sample_6stepx256x256/sampling_cfg_91b3cb8a95bc24d42b619fe2118fbf11_exp7.5_ddp_sample_6stepx256x256.json \
    working_dirSampleOutput/eval_step_fvd/exp7.5_to_gt 2


    /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256/sampling_cfg_66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256.json
    "/data/sample_outputs/66e009f115d466bfd2bc5930dc78d87c_exp6.2_ddp_sample_49x256x256",

comment