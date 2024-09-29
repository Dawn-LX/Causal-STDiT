

export CODE_ROOT="/home/gkf/project/CausalSTDiT"
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

torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/inference2.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR

<<comment

#################################
    configs/baselines/infer_example_SkyTimelapse_kv_cache.py \

    # causal-attn cyclic tpe  w/ kv-cache
    bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
    configs/_test_fps_for_causalSTDiT/SkyTimelapse_NoText_kv_cache.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
    working_dirSampleOutput/test_fps_NoTxt/causal_attn_with_kv_cache_MaxCondLen65 \
    9923 0

    # causal-attn cyclic tpe  w/o kv-cache
    bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
    configs/_test_fps_for_causalSTDiT/SkyTimelapse_NoText_No_kv_cache.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
    working_dirSampleOutput/test_fps_NoTxt/causal_attn_without_kv_cache_MaxCondLen65 \
    9923 0


    # full-attn fixed tpe
    bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
    configs/_test_fps_for_causalSTDiT/SkyTimelapse_NoText_No_kv_cache.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000 \
    working_dirSampleOutput/test_fps_NoTxt/full_attn_MaxCondLen65 \
    9786 0

comment