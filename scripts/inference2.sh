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
    scripts/inference2.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR

<<comment

############### 2024-11-07 for cvpr2025 ##################

# exp7  full-attn fixed tpe
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/training_config_backup.json2024-08-27T16-28-10.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/epoch3-global_step13000 \
    working_dirSampleOutput/exp7_fullattn_CfattnPp3_fixed_tpe33 \
    9985 0

# exp 7.2 full-attn fixed-TEP w/o cf-attn
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/training_config_backup.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/epoch2-global_step11000 \
    working_dirSampleOutput/exp7.2_fullattn_NoCfattn_fixed_tpe33_ppt50 \
    9185 2

# exp6 causal cyclic tpe  
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
    working_dirSampleOutput/exp6_pure_causal_CfattnPp3_tpe33_with_kvCache \
    9986 0

# exp6.2 causal cyclic-TPE w/o cf-attn
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 \
    working_dirSampleOutput/exp6.2_pure_causal_NoCfattn_tpe33_with_kvCache_ppt50 \
    9986 0

# exp6.4 causal cyclic-TPE w/o cf-attn maxCond=41,maxTPE=49
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/training_config_backup.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/epoch4-global_step12000 \
    working_dirSampleOutput/exp6.4_pure_causal_NoCfattn_tpe49_with_kvCache_ppt50 \
    9986 0
    
# exp6.5 causal cyclic-TPE w/ cf-attn maxCond=41,maxTPE=49
    bash scripts/inference2.sh \
    configs/ablations_infer_on_SkyTimelapse/infer_withKVcache_maxCond41.py \
    /data/CausalSTDiT_working_dir/exp6.5_pure_causal_CfattnPp3_tpe49/training_config_backup.json \
    /data/CausalSTDiT_working_dir/exp6.5_pure_causal_CfattnPp3_tpe49/epoch3-global_step11000 \
    working_dirSampleOutput/exp6.5_pure_causal_CfattnPp3_tpe49_with_kvCache_ppt50 \
    9986 0
    


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
        configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
        /data/CausalSTDiT_working_dir/exp4_ParitalCausal33x256x256ArSize8pp3_tpe33_timelapse_rmVidPadding/training_config_backup.json2024-08-24T21-45-05.json \
        /data/CausalSTDiT_working_dir/exp4_ParitalCausal33x256x256ArSize8pp3_tpe33_timelapse_rmVidPadding/epoch1-global_step11000 \
        working_dirSampleOutput/partialCausalCyclic_rmVidPadding_with_kvcache \
        9985 0
    
    # overfit partial causal w/o cf-attn
        bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
        configs/baselines/infer_example_overfit_beach.py \
        /data/CausalSTDiT_working_dir/exp4_overfit_25x256x256ArSize8NoCfAttn_tpe33/training_config_backup.json2024-08-14T21-48-57.json \
        /data/CausalSTDiT_working_dir/exp4_overfit_25x256x256ArSize8NoCfAttn_tpe33/epoch8-global_step13000 \
        working_dirSampleOutput/overfit_beach_MaxCond25_partialCausalCyclic_with_kv_cache \
        9985 0
    
###########################
    # exp7  full-attn fixed tpe
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
    /data/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/training_config_backup.json2024-08-27T16-28-10.json \
    /data/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/epoch3-global_step13000 \
    working_dirSampleOutput/exp7_fullattn_CfattnPp3_fixed_tpe33 \
    9985 0


    # exp5 partial causal cyclic tpe
    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_NoKVCache.py \
    /data/CausalSTDiT_working_dir/exp5_partial_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T17-33-43.json \
    /data/CausalSTDiT_working_dir/exp5_partial_causal_CfattnPp3_tpe33/epoch3-global_step14000 \
    working_dirSampleOutput/exp5_partial_causal_CfattnPp3_tpe33_without_kvCache \
    9988 0

    # exp9 pure causal rope

    bash scripts/inference2.sh \
    configs/baselines/infer_example_SkyTimelapse_kv_cache.py \
    /data/CausalSTDiT_working_dir/exp9_purecausal_CfattnPp3_rope33/training_config_backup.json2024-08-28T13-18-51.json \
    /data/CausalSTDiT_working_dir/exp9_purecausal_CfattnPp3_rope33/epoch4-global_step20000 \
    working_dirSampleOutput/exp9_purecausal_CfattnPp3_rope33_withKVCache \
    9981 0


#############

    # exp6 causal cyclic tpe
    bash scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapse.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
    working_dirSampleOutput/exp6_pure_causal_CfattnPp3_tpe33_with_kvCache \
    9986 2

    # exp8 partial causal rope
    bash scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapseWithKVCache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp8_partialcausal_CfattnPp3_rope33/training_config_backup.json2024-08-27T22-36-04.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp8_partialcausal_CfattnPp3_rope33/epoch3-global_step16000 \
    working_dirSampleOutput/exp8_partialcausal_CfattnPp3_rope33_with_kvCache \
    9981 1

    # exp10 full attn rope
    bash scripts/inference2.sh \
    configs/causal_stdit/infer_example_SkyTimelapseWithoutKVCache.py \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp10_fullattn_CfattnPp3_rope33_save_optim/training_config_backup.json \
    /data9T/gaokaifeng/CausalSTDiT_working_dir/exp10_fullattn_CfattnPp3_rope33_save_optim/epoch4-global_step20000 \
    working_dirSampleOutput/exp10_fullattn_CfattnPp3_rope33 \
    9983 0

comment