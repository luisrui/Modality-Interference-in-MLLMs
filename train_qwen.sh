export PYTHONPATH=~/MLLM:$PYTHONPATH

# Qwen2.5-VL + FFT
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7" src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-7b/vinilla/args_onlyvqa.yaml

# Qwen2.5-VL + KL
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7 " src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-7b/only_consistency/args_KL_full.yaml

# Qwen2.5-VL + JS
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7 " src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-7b/only_consistency/args_JS_full.yaml

# Qwen2.5-VL + RG
CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7" src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-7b/only_adversarial/args_full_RG.yaml

# Qwen2.5-VL + PGD
# CUDA_VISIBLE_DEVICES=6,7  deepspeed --master_port=29522 --include="localhost:6,7 " src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-3b/only_adversarial/args_full_PGD.yaml

# Qwen2.5-VL + PGD + KL
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7 " src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-3b/args_full_KL_PGD.yaml

# Qwen2.5-VL + PGD + JS
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port=29522 --include="localhost:4,5,6,7 " src/qwen/qwen_consistent.py --config configs/model_train/qwen2.5-vl-7b/args_full_JS_PGD.yaml