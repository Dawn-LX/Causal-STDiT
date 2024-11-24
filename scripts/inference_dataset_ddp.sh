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


ABS_CFG_PATH=${1}
ABS_TRAIN_CFG=${2}
ABS_CKPT_PATH=${3}
EXP_DIR=${4}
MASTER_PORT=${5}
export CUDA_VISIBLE_DEVICES=${6}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export IS_DEBUG=0
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/inference_dataset_ddp.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR \
    --sample_save_dir $SAMPLE_SAVE_DIR

<<comment

# debug


    bash scripts/inference_dataset_ddp.sh \
    configs/ddp_sample_skytimelapse/chunk8_MaxCond41_ArSteps6_withKVcache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/training_config_backup.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/epoch3-global_step11000 \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4ckpt11k_ddpsample_49x256x256 \
    9977 2

    
comment