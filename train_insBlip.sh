export PYTHONPATH=~/MLLM:$PYTHONPATH

# InstructBlip + FFT
# CUDA_VISIBLE_DEVICES=6 deepspeed --master_port=29514 --include="localhost:7" src/instructblip/insBlip_consistent.py --config configs/model_train/instructBlip-7b/vinilla/args_full.yaml

# InstructBlip + FFT + noVQA data
# CUDA_VISIBLE_DEVICES=4,5 deepspeed --master_port=29517 --include="localhost:4,5" src/instructblip/insBlip_consistent.py --config configs/model_train/instructBlip-7b/vinilla/args_full_novqa.yaml

# InstructBlip + FFT + onlyvqa
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29518 --include="localhost:4,5,6,7" src/instructblip/insBlip_consistent.py --config configs/model_train/instructBlip-7b/vinilla/args_full_nounimodality.yaml

# InstructBlip + FFT + KL
# CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29517 --include="localhost:4,5,6,7" src/instructblip/insBlip_consistent.py --config configs/model_train/instructBlip-7b/only_consistency/args_JS_full.yaml

# InstructBlip + FFT + PGD
# CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29519 --include="localhost:4,5,6,7" src/instructblip/insBlip_consistent.py --config configs/model_train/instructBlip-7b/args_full_KL_PGD.yaml