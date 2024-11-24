if test -d "/data"; then
    # for server 10.130.129.11
    export CODE_ROOT="/home/gkf/project/CausalSTDiT"
    
    export ROOT_CKPT_DIR="/home/gkf/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data"
    export ROOT_OUTPUT_DIR="/data/CausalSTDiT_working_dir"
    echo "on server A100"
else
    # for server 10.130.129.34
    export CODE_ROOT="/data9T/gaokaifeng/project/CausalSTDiT"

    export ROOT_CKPT_DIR="/data9T/gaokaifeng/LargeModelWeightsFromHuggingFace"
    export ROOT_DATA_DIR="/data9T/gaokaifeng/datasets"
    export ROOT_OUTPUT_DIR="/data9T/gaokaifeng/CausalSTDiT_working_dir"
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

export IS_DEBUG=1
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
export DEBUG_KV_CACHE2=0
export DEBUG_KV_CACHE3=0
# export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debug_KVcache_wo_CfAttn"
# export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debug_KVcache"
export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25"
export KV_CACHE_TAG=${EXP_DIR:${#EXP_DIR}-15}

torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/inference.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR

<<comment


### pp_t=50

    # exp6 maxCOnd=25
        bash scripts/inference.sh \
        configs/ablations_infer_on_SkyTimelapse/infer_withKVcache_maxCond25.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/exps_prefix_perturb_t50/exp6_maxCond25 \
        9981 1

    # exp6.2 maxCond=25
        bash scripts/inference.sh \
        configs/ablations_infer_on_SkyTimelapse/infer_withKVcache_maxCond25.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/exps_prefix_perturb_t50/exp6.2_maxCond25 \
        9982 2


comment