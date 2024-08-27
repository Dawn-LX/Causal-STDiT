

if test -d "/data"; then
    # for server 10.130.129.11
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    
    export ROOT_CKPT_DIR="/home/gkf/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data"
    export ROOT_OUTPUT_DIR="/data/CausalSTDiT_working_dir"
else
    # for server 10.130.129.34
    export CODE_ROOT="/data9T/gaokaifeng/project/CausalSTDiT"

    export ROOT_CKPT_DIR="/data9T/gaokaifeng/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data9T/gaokaifeng/datasets"
    export ROOT_OUTPUT_DIR="/data9T/gaokaifeng/CausalSTDiT_working_dir"
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

outputs = f"/data/CausalSTDiT_working_dir/exp4_ParitalCausal{_exp_tag}pp3_tpe33_timelapse_rmVidPadding"

_ROOT_CKPT_DIR = os.getenv("ROOT_CKPT_DIR","/home/gkf/LargeModelWeightsFromHuggingFace") # or /data9T/gaokaifeng/LargeModelWeightsFromHuggingFace
_ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #
# /data/SkyTimelapse or /data9T/gaokaifeng/datasets/SkyTimelapse

# debug overfit

    bash scripts/train.sh \
    configs/causal_stdit/overfit_beach_25x256x256_ar8.py \
    9686 0

# train skytimelapse

    bash scripts/train.sh \
    configs/causal_stdit/train_SkyTimelapse_33x256x256_TPE33.py \
    9686 0

# train partial-causal:

    # overfit-beach
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/causal_stdit/overfit_beach_ParitalCausal_CyclicTpe33.py \
        9686 0

    # train skyline timelapse
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/baselines/exp4_partialcausal_attn_cyclic_tpe33.py \
        9686 0
    
# ablations on  skytimelapse:

    # full-attn fixed tpe
    bash scripts/train.sh \
    TODO \
    9686 0


    # partial causal attn cyclic tpe
    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp5_partialcausal_attn_cyclic_tpe33.py \
    exp5_partial_causal_CfattnPp3_tpe33 \
    9686 0

    # causla attn cyclic tpe
    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp6_purecausal_attn_cyclic_tpe33.py \
    exp6_pure_causal_CfattnPp3_tpe33
    9686 1
comment