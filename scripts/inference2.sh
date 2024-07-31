export CODE_ROOT="/home/gkf/project/CausalSTDiT"
export PYTHONPATH=$PYTHONPATH:$CODE_ROOT

cd $CODE_ROOT
pwd


ABS_CFG_PATH=${1}
ABS_CKPT_PATH=${2}
MASTER_PORT=${3}
export CUDA_VISIBLE_DEVICES=${4}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export IS_DEBUG=0
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/inference2.py \
    --config $ABS_CFG_PATH \
    --ckpt_path $ABS_CKPT_PATH

<<comment

# debug

/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/training_config_backup.json
/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/epoch0-global_step500

scripts/inference2.sh \
/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/training_config_backup.json \
/data/CausalSTDiT_working_dir/25x256x256fi3ArSize8_overfit_beach/epoch0-global_step500 \
 9686 0

comment