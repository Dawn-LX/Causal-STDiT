
export CODE_ROOT="/home/gkf/project/CausalSTDiT"
export PYTHONPATH=$PYTHONPATH:$CODE_ROOT

cd $CODE_ROOT


torchrun \
    --nnodes=1 \
    --master-port=6988 \
    --nproc-per-node=1 \
    tests/test_dataloader.py