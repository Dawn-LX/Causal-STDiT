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
# export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debug_KVcache_wo_CfAttn"
# export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debug_KVcache"
export TENSOR_SAVE_DIR="/home/gkf/project/CausalSTDiT/working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25"
export KV_CACHE_TAG=${EXP_DIR:${#EXP_DIR}-15}

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

parser.add_argument("--exp_dir",type=str, default="/data/CausalSTDiT_working_dir/debug")
    parser.add_argument("--sample_save_dir",type=str, default="/data/sample_outputs")

# debug

### overfit-beach

    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    /home/gkf/project/CausalSTDiT/configs/causal_stdit/infer_example_beach.py \
    /data/CausalSTDiT_working_dir/depth14_25x256x256fi3ArSize8_overfit_beach/training_config_backup.json \
    /data/CausalSTDiT_working_dir/depth14_25x256x256fi3ArSize8_overfit_beach/epoch9-global_step14500 \
    working_dirSampleOutput/overfit_beach_wo_kv_cache \
    9987 0

### SkyTimelapse 33x256x256ArSize8pp3
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    /home/gkf/project/CausalSTDiT/configs/causal_stdit/infer_example_SkyTimelapse.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
    working_dirSampleOutput/33x256x256ArSize8pp3_14k_v2 \
    9986 0

### SkyTimelapse ablations

    # full-attn fixed tpe ckpt-10k
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    /home/gkf/project/CausalSTDiT/configs/baselines/infer_exp1.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000 \
    working_dirSampleOutput/ablations/FullAttnFixTpe_maxCondLen25 \
    9988 0

    # causal attn cyclic tpe ckpt-10k
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapse.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step10000 \
    working_dirSampleOutput/ablations/CausalAttnCyclicTpe_maxKVcache25 \
    9986 0

    # causal attn cyclic tpe ckpt-14k
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapse.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
    working_dirSampleOutput/ablations/CausalAttnCyclicTpe_maxKVcache25_14k \
    9986 0

#################################

    # causal-attn cyclic-tpe
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step13000
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000

    # causal-attn fixed tpe
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
    working_dirSampleOutput/CausalFixed_maxKVcache33_13k \
    9993 0

    # full-attn fixed tpe
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json

    # full-attn cyclic tpe
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapse.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/training_config_backup.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
    working_dirSampleOutput/exp3_full_attn_cyclic_tpe64_ckpt13k_MaxCondLen33 \
    9786 0

    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse.py \
    TODO TODO TODO \
    9187 0

    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    _backup/infer_example_SkyTimelapse0.py \
    TODO TODO TODO \
    9986 0

########### debug reorganized tpe

    full-attn fixed tpe w/o kv-cache
    full-attn cyclic tpe w/o kv-cache
    
    causal-attn fixed tpe w/o kv-cache
    causal-attn cyclic tpe w/o kv-cache
    causal-attn fixed tpe w/ kv-cache
    causal-attn cyclic tpe w/ kv-cache

    ### full-attn fixed tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/FullFixed_wo_kv_cache \
        9981 0

    ### full-attn cyclic tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/training_config_backup.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/FullCyclic_wo_kv_cache \
        9982 0
    
    ### causal-attn fixed tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalFixed_wo_kv_cache \
        9983 0

    ### causal-attn cyclic tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalCyclic_wo_kv_cache \
        9984 0
    
    ### causal-attn fixed tpe w/ kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalFixed_with_kv_cache \
        9985 0

    ### causal-attn cyclic tpe w/ kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalCyclic_with_kv_cache \
        9986 0
    
####### debug w/ & w/o kv-cache

    ### causal-attn fixed tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
        working_dirSampleOutput/_debug_KVcache/CausalFixedMaxCond25_wo_kv_cache \
        9983 0
    
    ### causal-attn fixed tpe w/ kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000 \
        working_dirSampleOutput/_debug_KVcache/CausalFixedMaxCond25_with_kv_cache \
        9985 0
    

    ### causal-attn cyclic tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalCyclic_wo_kv_cache \
        9983 0
    
    ### causal-attn cyclic tpe w/ kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
        /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
        working_dirSampleOutput/_debugReorganizedTpeMaxCondLen25/CausalCyclic_with_kv_cache \
        9985 0

#
########## debug partial causal

    ### causal-attn cyclic tpe w/o kv-cache
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
        /data/CausalSTDiT_working_dir/exp4_ParitalCausal33x256x256ArSize8pp3_tpe33_timelapse/training_config_backup.json \
        /data/CausalSTDiT_working_dir/exp4_ParitalCausal33x256x256ArSize8pp3_tpe33_timelapse/epoch2-global_step18000 \
        working_dirSampleOutput/partialCausalCyclic_NoisePadding_wo_cache \
        9985 0
    
    # overfit partial causal w/o cf-attn
    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    configs/baselines/infer_example_overfit_beach.py \
    /data/CausalSTDiT_working_dir/exp4_overfit_25x256x256ArSize8NoCfAttn_tpe33/training_config_backup.json2024-08-14T21-48-57.json \
    /data/CausalSTDiT_working_dir/exp4_overfit_25x256x256ArSize8NoCfAttn_tpe33/epoch8-global_step13000 \
    working_dirSampleOutput/overfit_beach_MaxCond25_partialCausalCyclic_with_kv_cache \
    9985 0
    





comment