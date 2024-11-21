

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

########## test fps 2024-11-13

    # exp6 w/ txt, w/ cf-attn, maxCond=25

    bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
    configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_kv_cache.py \
    /data/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
    /data/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
    working_dirSampleOutput/test_fps_withTxt/exp6_maxCond25 \
    9786 0

    num_gen_frames=160, time_used=131.56, fps=1.22
    num_gen_frames=120, time_used=98.71, fps=1.22
    num_gen_frames=80, time_used=65.98, fps=1.21
    

    # exp6.2 w/ txt, w/o cf-attn, maxCond=25

        bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_kv_cache.py \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/test_fps_withTxt/exp6.2_maxCond25 \
        9786 0
        
        num_gen_frames=80, time_used=52.08, fps=1.54

    # exp6.4 w/ txt w/o cf-attn, maxCond=41
        bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_kv_cache.py \
        /data/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/training_config_backup.json \
        /data/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/epoch3-global_step11000 \
        working_dirSampleOutput/test_fps_withTxt/exp6.4_maxCond41 \
        9781 0

        num_gen_frames=80, time_used=53.58, fps=1.49

    # exp7.2 w/ txt, w/o cf-attn, maxCond=25

        bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache.py \
        /data/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/training_config_backup.json \
        /data/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/epoch2-global_step11000 \
        working_dirSampleOutput/test_fps_withTxt/exp7.2_maxCond25 \
        9786 0

        num_gen_frames=80, time_used=130.09, fps=0.61

    # exp7.3 w/ txt w/o cf-attn maxCond=41
        bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache.py \
        /data/CausalSTDiT_working_dir/exp7.3_fullattn_NoCfattn_fixed_tpe49/training_config_backup.json2024-11-10T17-13-50.json \
        /data/CausalSTDiT_working_dir/exp7.3_fullattn_NoCfattn_fixed_tpe49/epoch3-global_step11000 \
        working_dirSampleOutput/test_fps_withTxt/exp7.3_maxCond41 \
        9786 0
        
        num_gen_frames=80, time_used=167.33, fps=0.48

    # exp7.4 w/ txt w/o cf-attn, fixCond=8

        bash /home/gkf/project/CausalSTDiT/scripts/inference_test_fps.sh \
        configs/_test_fps_for_causalSTDiT/SkyTimelapse_withText_No_kv_cache_given8.py \
        /data/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/training_config_backup.json2024-11-10T22-11-11.json \
        /data/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/epoch1-global_step11000 \
        working_dirSampleOutput/test_fps_withTxt/exp7.7_fixcond8 \
        9786 0
        
        num_gen_frames=80, time_used=77.54, fps=1.03


comment