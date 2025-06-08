export PYTHONPATH=~/MLLM:$PYTHONPATH

# LLaVA + FFT
# CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29520 --include="localhost:4,5,6,7" src/llava/llava_consistent.py --config configs/model_train/llava-v1.5-7b/vinilla/args_full.yaml

# LLaVA + Lora + SFT
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29520 --include="localhost:4,5,6,7" src/llava/llava_consistent.py --config configs/model_train/llava-v1.5-13b/vinilla/args_lora.yaml

# LLaVA + KL
# CUDA_VISIBLE_DEVICES=4,5,6 deepspeed --master_port=29520 --include="localhost:4,5,6" src/llava/llava_consistent.py --config configs/model_train/llava-v1.5-7b/only_consistency/args_KL_full.yaml

# LLaVA + PGD
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29520 --include="localhost:4,5,6,7 " src/llava/llava_consistent.py --config configs/model_train/llava-v1.5-7b/only_adversarial/args_full_PGD.yaml

# LLaVA + PGD + KL
# CUDA_VISIBLE_DEVICES=4,5,6 deepspeed --master_port=29520 --include="localhost:4,5,6" src/llava/llava_consistent.py --config configs/model_train/llava-v1.5-7b/args_full_KL_PGD.yaml 