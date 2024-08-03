export CODE_ROOT="/home/gkf/project/CausalSTDiT"
export PYTHONPATH=$PYTHONPATH:$CODE_ROOT

cd $CODE_ROOT
pwd

ABS_SAMPLE_CONFIG=${1}
EXP_DIR=${2}
export CUDA_VISIBLE_DEVICES=${3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export I3D_WEIGHTS_DIR="/home/gkf/project/CausalSTDiT/_backup/common_metrics_on_video_quality-main/fvd"
# i3d_weights_path = f"{I3D_WEIGHTS_DIR}/styleganv/i3d_torchscript.pt" (for styleganv)
# i3d_weights_path = f"{I3D_WEIGHTS_DIR}/videogpt/i3d_pretrained_400.pt" (for videogpt)

python scripts/eval_fvd.py \
    --sample_config $ABS_SAMPLE_CONFIG \
    --exp_dir $EXP_DIR \
    --batch_size 6 \
    --num_workers 4


<<comment


bash /home/gkf/project/CausalSTDiT/scripts/eval_fvd.sh \
/data/CausalSTDiT_working_dir/debug_inference/sampling_cfg_14831e3e05cfd0b1d0a97c2ff2a6b3f5_debug_inference.json \
working_dirSampleOutput/eval_fvd 3

comment