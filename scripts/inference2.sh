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

export IS_DEBUG=0
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
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

### SkyTimelapse full attn fix tpe baseline

    bash /home/gkf/project/CausalSTDiT/scripts/inference2.sh \
    /home/gkf/project/CausalSTDiT/configs/baselines/infer_exp1.py \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json \
    /data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000 \
    working_dirSampleOutput/exp1_full_attn_fix_tpe \
    9988 0


comment