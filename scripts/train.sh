

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


CFG_PATH=${1}
EXP_NAME=${2}
MASTER_PORT=${3}
export CUDA_VISIBLE_DEVICES=${4}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


export IS_DEBUG=0
export DEBUG_COND_LEN=1
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
export TOKENIZERS_PARALLELISM=false
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/train.py \
    --config $CODE_ROOT/$CFG_PATH \
    --outputs $ROOT_OUTPUT_DIR/$EXP_NAME

<<comment

outputs = f"/data/CausalSTDiT_working_dir/exp4_ParitalCausal{_exp_tag}pp3_tpe33_timelapse_rmVidPadding"

_ROOT_CKPT_DIR = os.getenv("ROOT_CKPT_DIR","/home/gkf/LargeModelWeightsFromHuggingFace") # or /data9T/gaokaifeng/LargeModelWeightsFromHuggingFace
_ROOT_DATA_DIR = os.getenv("ROOT_DATA_DIR","/data")  #
# /data/SkyTimelapse or /data9T/gaokaifeng/datasets/SkyTimelapse

# debug overfit

    bash scripts/train.sh \
    configs/causal_stdit/overfit_beach_25x256x256_ar8.py \
    9686 0

# train skytimelapse

    bash scripts/train.sh \
    configs/causal_stdit/train_SkyTimelapse_33x256x256_TPE33.py \
    9686 0

# train partial-causal:

    # overfit-beach
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/causal_stdit/overfit_beach_ParitalCausal_CyclicTpe33.py \
        9686 0

    # train skyline timelapse
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/baselines/exp4_partialcausal_attn_cyclic_tpe33.py \
        9686 0
    
# ablations on  skytimelapse:

    #exp7 full-attn fixed tpe
        # w/ cf-attn
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp7_fullattn_fixed_tpe33.py \
        exp7_fullattn_CfattnPp3_fixed_tpe33 \
        9686 0

        # w/o Cf-Attn
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp7.2_fullattn_fixed_tpe33_NoCfAttn.py \
        exp7.2_fullattn_NoCfattn_fixed_tpe33 \
        9686 0

        # w/o Cf-Attn maxCond41-tpe49
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp7.3_fullattn_fixed_tpe49_NoCfAttn.py \
        exp7.3_fullattn_NoCfattn_fixed_tpe49 \
        9681 1

        # exp 7.4 full-attn fixed-TEP w/o cf-attn, w/o extendable condition
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp7.4_fullattn_fixed_tpe16_NoCfAttn.py \
        exp7.4_fullattn_NoCfattn_fixed_tpe16 \
        9081 0

        # exp 7.5 full-attn fixed-TEP w/o cf-attn, w/o extendable condition, cond_len = [1,8]
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp7.5_fullattn_fixed_tpe16_NoCfAttn.py \
        exp7.5_fullattn_NoCfattn_Cond1and8_fixed_tpe16 \
        9581 0


    # partial causal attn cyclic tpe
    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp5_partialcausal_attn_cyclic_tpe33.py \
    exp5_partial_causal_CfattnPp3_tpe33 \
    9686 0

    # exp6 pure causal attn cyclic tpe
        
        # w/ cf-attn
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp6_purecausal_attn_cyclic_tpe33.py \
        exp6_pure_causal_CfattnPp3_tpe33 \
        9686 1

        # w/o cf-attn
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp6.2_resume.py \
        exp6.2_pure_causal_NoCfattn_tpe33_Fromckpt11k \
        9681 2



        # w/o cf-attn maxCond65-tpe73
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp6.3_purecausal_attn_cyclic_tpe73_NoCfattn.py \
        exp6.3_pure_causal_NoCfattn_tpe73 \
        9686 0
        

        # w/o cf-attn maxCond41 maxTPE49
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp6.4_purecausal_attn_cyclic_tpe49_NoCfattn.py \
        exp6.4_pure_causal_NoCfattn_tpe49 \
        9686 0

        # exp6.5  w/ cf-attn Pp3 maxCond41 maxTPE49
        bash scripts/train.sh \
        configs/ablations_on_SkyTimelapse/exp6.5_purecausal_attn_cyclic_tpe49_CfattnPp3.py \
        exp6.5_pure_causal_CfattnPp3_tpe49 \
        9686 0

    # pure causal  RoPE
    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp9_purecausal_attn_rope33.py \
    exp9_purecausal_CfattnPp3_rope33 \
    9686 1

    # full attn RoPE
    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp10_fullattn_rope33.py \
    exp10_fullattn_CfattnPp3_rope33 \
    9197 2

    bash scripts/train.sh \
    configs/ablations_on_SkyTimelapse/exp8_partialcausal_attn_rope33.py \
    exp8_partialcausal_CfattnPp3_rope33 \
    9686 1
comment