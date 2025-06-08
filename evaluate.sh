#!/bin/bash
export PYTHONPATH=~/MLLM:$PYTHONPATH

# Check if the number of arguments is less than 2
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --model MODEL --tag TAG_NAME"
    echo "Example: $0 --model llava-1.5-7b --tag _Anchor_VQA_trail_3"
    exit 1
fi

# parser arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$TAG" ]; then
    echo "Both --model and --tag parameters are required"
    exit 1
fi

datasets=(
    "scienceqa_VQA"
    "mmbench-en_VQA"
    "seed-bench-img_VQA"
    "mmlu_text_heavy"
    "openbookqa_text_heavy"
    "caltech-101_image_heavy"
    "mini-imagenet_image_heavy"
)

# Accessment
for dataset in "${datasets[@]}"; do
    echo "Evaluating dataset: $dataset"
    python src/batch_evaluate.py \
        --data "$dataset" \
        --model "$MODEL" \
        --tag "$TAG"
done

echo "All evaluations completed!"