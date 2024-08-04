export CODE_ROOT="/home/gkf/project/CausalSTDiT"
export PYTHONPATH=$PYTHONPATH:$CODE_ROOT

cd $CODE_ROOT
pwd


CFG_PATH=${1}
# EXP_NAME=${2}
MASTER_PORT=${2}
export CUDA_VISIBLE_DEVICES=${3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export IS_DEBUG=0
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
export TOKENIZERS_PARALLELISM=false
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/train.py \
    --config $CODE_ROOT/$CFG_PATH

<<comment

# debug overfit

    bash scripts/train.sh \
    configs/causal_stdit/overfit_beach_25x256x256_ar8.py \
    9686 0

# train skytimelapse

    bash scripts/train.sh \
    configs/causal_stdit/train_SkyTimelapse_33x256x256_without_text_cond.py \
    9686 0

# train baseline:

    # full-attn fixed tpe
    bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
    configs/baselines/full_attn_fixed_tpe.py \
    9686 0

    


configs/causal_stdit/train_demo_65x256x256.py

git filter-branch --tree-filter 'rm -f Open-Sora-1.2.0.zip' HEAD
comment