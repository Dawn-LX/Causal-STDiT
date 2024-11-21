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
    scripts/inference_dataset_ddp2.py \
    --config $ABS_CFG_PATH \
    --train_config $ABS_TRAIN_CFG \
    --ckpt_path $ABS_CKPT_PATH \
    --exp_dir $EXP_DIR \
    --sample_save_dir $SAMPLE_SAVE_DIR

<<comment

parser.add_argument("--exp_dir",type=str, default="/data/CausalSTDiT_working_dir/debug")
    parser.add_argument("--sample_save_dir",type=str, default="/data/sample_outputs")

# debug

/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/training_config_backup.json
/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/epoch0-global_step500

bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh \
configs/causal_stdit/infer_dataset_SkyTimelapse.py \
/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json \
/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000 \
/data/CausalSTDiT_working_dir/debug_inference \
 9986 0


#### exp7 full-attn fixed tpe33 (new version, i.e., use iddpm rather than CleanPrefixIDDPM, and use inference_dataset_ddp2.py)

bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh \
configs/baselines/infer_dataset_SkyTimelapse.py \
/data/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/training_config_backup.json2024-08-27T16-28-10.json \
/data/CausalSTDiT_working_dir/exp7_fullattn_CfattnPp3_fixed_tpe33/epoch3-global_step13000 \
/data/CausalSTDiT_working_dir/exp7_ddp_sample_17x256x256 \
 9986 0


### exp6 pure causal fixed tpe33

bash scripts/inference_dataset_ddp.sh \
configs/baselines/infer_dataset_SkyTimelapse.py \
/data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
/data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
/data9T/gaokaifeng/CausalSTDiT_working_dir/exp6_ddp_sample_17x256x256 \
 9986 2


######## 2024-11-10 for cvpr2025
    # exp6 causal cyclic-TPE w/ cf-attn maxCond=25,maxTPE=33

    bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh \
    /home/gkf/project/CausalSTDiT/configs/ddp_sample_skytimelapse/chunk8_MaxCond25_ArSteps6_withKVcache.py \
    /data/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/training_config_backup.json2024-08-26T19-34-35.json \
    /data/CausalSTDiT_working_dir/exp6_pure_causal_CfattnPp3_tpe33/epoch2-global_step11000 \
    /data/CausalSTDiT_working_dir/exp6CfAttnpp3_ddp_sample_49x256x256 \
    9977 0

    # exp6.2 causal cyclic-TPE w/o cf-attn maxCond=25,maxTPE=33
        bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh \
        /home/gkf/project/CausalSTDiT/configs/ddp_sample_skytimelapse/chunk8_MaxCond25_ArSteps6_withKVcache.py \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 \
        /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256 \
        9977 0
    
    # exp 6.2 11k+20k
        bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh \
        /home/gkf/project/CausalSTDiT/configs/ddp_sample_skytimelapse/chunk8_MaxCond25_ArSteps6_withKVcache.py \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33_Fromckpt11k/training_config_backup.json \
        /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33_Fromckpt11k/epoch4-global_step20000 \
        /data/CausalSTDiT_working_dir/exp6.2_31k_ddpsample49x256x256 \
        9977 0

    crontab:
    46 22 * * 1 bash /home/gkf/project/CausalSTDiT/scripts/inference_dataset_ddp.sh /home/gkf/project/CausalSTDiT/configs/ddp_sample_skytimelapse/chunk8_MaxCond25_ArSteps6_withKVcache.py /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/training_config_backup.json2024-09-30T15-54-59.json /data/CausalSTDiT_working_dir/exp6.2_pure_causal_NoCfattn_tpe33/epoch2-global_step11000 /data/CausalSTDiT_working_dir/exp6.2_ddp_sample_49x256x256 9971 0

    # exp6.4 causal cyclic-TPE w/o cf-attn maxCond=41,maxTPE=49

        bash scripts/inference_dataset_ddp.sh \
        configs/ddp_sample_skytimelapse/chunk8_MaxCond41_ArSteps6_withKVcache.py \
        /data/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/training_config_backup.json \
        /data/CausalSTDiT_working_dir/exp6.4_pure_causal_NoCfattn_tpe49/epoch4-global_step12000 \
        /data/CausalSTDiT_working_dir/exp6.4_ddp_sample_49x256x256 \
        9966 0

        # bsz=4 per gpu, 2gpu, 140s/it
    
    # exp 7.2 full-attn fixed-TEP w/o cf-attn maxCond=25, maxTPE=33

        bash scripts/inference_dataset_ddp.sh \
        configs/ddp_sample_skytimelapse/chunk8_MaxCond25_ArSteps6_NoKVcache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/training_config_backup.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_fullattn_NoCfattn_fixed_tpe33/epoch2-global_step11000 \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.2_ddp_sample_49x256x256 \
        9766 2

    # exp 7.4 full-attn fixed-TEP w/o cf-attn fixedCond=8, TPE=16
        bash scripts/inference_dataset_ddp.sh \
        configs/ddp_sample_skytimelapse/chunk8_FixCond8_ArSteps6_NoKVcache.py \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/training_config_backup.json2024-11-10T22-11-11.json \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_fullattn_NoCfattn_fixed_tpe16/epoch1-global_step11000 \
        /data9T/gaokaifeng/CausalSTDiT_working_dir/exp7.4_ddp_sample_6stepx256x256 \
        9766 2


comment