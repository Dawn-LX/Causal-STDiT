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

comment