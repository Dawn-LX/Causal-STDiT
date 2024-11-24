if test -d "/data"; then
    # for server A100
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    
    export ROOT_CKPT_DIR="/home/gkf/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data"
    export SAMPLE_SAVE_DIR="/data/sample_outputs"
    echo "on server A100"
else
    # for server A6000
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


## example

    bash scripts/eval_step_fvd.sh \
    /path/to/sampling_cfg_backup.json \
    working_dirSampleOutput/eval_step_fvd 0

comment