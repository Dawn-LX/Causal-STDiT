

if test -d "/data"; then
    # 
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    
    export ROOT_CKPT_DIR="/home/gkf/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data"
    export ROOT_OUTPUT_DIR="/data/CausalSTDiT_working_dir"
    echo "on server A100"
else
    export CODE_ROOT="/data9T/gaokaifeng/project/CausalSTDiT"

    export ROOT_CKPT_DIR="/data9T/gaokaifeng/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data9T/gaokaifeng/datasets"
    export ROOT_OUTPUT_DIR="{$CODE_ROOT}/working_dir"
    echo "on server A6000"
fi

export PYTHONPATH=$PYTHONPATH:$CODE_ROOT
cd $CODE_ROOT
pwd


CFG_PATH=${1}
EXP_NAME=${2}
MASTER_PORT=${3}
export CUDA_VISIBLE_DEVICES=${4}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


export IS_DEBUG=0
export DEBUG_COND_LEN=1
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
export TOKENIZERS_PARALLELISM=false
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/train.py \
    --config $CODE_ROOT/$CFG_PATH \
    --outputs $ROOT_OUTPUT_DIR/$EXP_NAME

<<comment


# debug overfit

    bash scripts/train.sh \
    configs/causal_stdit/train_overfit_beach_demo.py \
    debug_overfit \
    9686 0

# train skytimelapse

    bash scripts/train.sh \
    configs/causal_stdit/train_SkyTimelapse_33x256x256_TPE33.py \
    9686 0

comment