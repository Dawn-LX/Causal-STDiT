
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
export FPS_INFO_SAVE_DIR=$EXP_DIR
export DEBUG_TURNOFF_XFORMERS=1

torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/inference2_estimate_flops.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR

<<comment



#############
    # w/ text_encoder T5

    # Baseline-Ext, cond25 w/o PE
        bash scripts/inference2_estimate_flops.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/training_config_backup.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/test_flops_withTxt/exp7.2_BaselineExt_maxCond25 \
        9982 2
    
    # Baseline-Ext, cond41 w/o PE

        bash scripts/inference2_estimate_flops.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.3_fullattn_NoCfattn_fixed_tpe49/training_config_backup.json2024-11-10T17-13-50.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.3_fullattn_NoCfattn_fixed_tpe49/epoch3-global_step11000 \
        working_dirSampleOutput/test_flops_withTxt/exp7.3_maxCond41 \
        9953 1

    # Baseline-Fix, cond8 w/o PE

        bash scripts/inference2_estimate_flops.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache_given8.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/training_config_backup.json2024-11-10T22-11-11.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/epoch1-global_step11000 \
        working_dirSampleOutput/test_flops_withTxt/exp7.4_fixCond8_given8 \
        9984 1

    
    # Ours, cond25 w/o PE
        bash scripts/inference2_estimate_flops.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_kv_cache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/test_flops_withTxt/exp6.2_maxCond25 \
        9980 0
    
    # Ours, Cond41 w/o PE

        bash scripts/inference2_estimate_flops.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_kv_cache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/training_config_backup.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/epoch3-global_step11000 \
        working_dirSampleOutput/test_flops_withTxt/exp6.4_maxCond41 \
        9981 2

comment