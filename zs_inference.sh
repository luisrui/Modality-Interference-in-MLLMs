#!/bin/bash

export PYTHONPATH=~/MLLM:$PYTHONPATH

gpu_ids=(4 5 6 7)

# define tasks
declare -A tasks
tasks["VQA-1"]="configs/model_inference/VQA/args_scienceqa.yaml"
tasks["VQA-2"]="configs/model_inference/VQA/args_mmbench.yaml"
tasks["VQA-3"]="configs/model_inference/VQA/args_seedbench.yaml"
tasks["Image-Heavy-1"]="configs/model_inference/image_heavy/args_caltech101.yaml"
tasks["Image-Heavy-2"]="configs/model_inference/image_heavy/args_miniImageNet.yaml"
tasks["Text-Heavy-1"]="configs/model_inference/text_heavy/args_mmlu.yaml"
tasks["Text-Heavy-2"]="configs/model_inference/text_heavy/args_openbookqa.yaml"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --checkpoint_path)
            checkpoint_path="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --tag)
            tag="$2"
            shift 2
            ;;
        --all)
            all=true
            shift
            ;;
        --text_heavy)
            shift
            text_heavy=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                text_heavy+=("$1")
                shift
            done
            ;;
        --image_heavy)
            shift
            image_heavy=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                image_heavy+=("$1")
                shift
            done
            ;;
        --vqa)
            vqa=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$model_name" ] || [ -z "$checkpoint_path" ] || [ -z "$batch_size" ] || [ -z "$tag" ]; then
    echo "Usage: $0 --model_name <model_name> --checkpoint_path <checkpoint_path> --batch_size <batch_size> --tag <tag> [--all] [--text_heavy sample1 sample2] [--image_heavy sample1 sample2] [--vqa]"
    exit 1
fi

text_heavy_samples=('random' 'switch' 'full_black' 'full_white')
image_heavy_samples=('related_text' 'origin' 'unrelated_text')

if [ "$all" == true ]; then
    selected_text_heavy_samples=("${text_heavy_samples[@]}")
    selected_image_heavy_samples=("${image_heavy_samples[@]}")
    vqa=true
else
    selected_text_heavy_samples=("${text_heavy[@]}")
    selected_image_heavy_samples=("${image_heavy[@]}")
fi

run_task() {
    local task_name=$1
    local config=$2

    echo "Running $task_name task with config: $config"
    for idx in "${!gpu_ids[@]}"; do
        gpu_id=${gpu_ids[$idx]}

        CMD="CUDA_VISIBLE_DEVICES=$gpu_id python src/predict.py \
            --config \"$config\" \
            --model_name \"$model_name\" \
            --checkpoint_path \"$checkpoint_path\" \
            --batch_size \"$batch_size\" \
            --tag \"$tag\" \
            --num-chunks \"${#gpu_ids[@]}\" \
            --chunk-idx \"$idx\""

        if [[ ${#selected_text_heavy_samples[@]} -gt 0 ]]; then
            CMD+=" --text_heavy ${selected_text_heavy_samples[@]}"
        fi

        if [[ ${#selected_image_heavy_samples[@]} -gt 0 ]]; then
            CMD+=" --image_heavy ${selected_image_heavy_samples[@]}"
        fi

        if [[ "$vqa" == true ]]; then
            CMD+=" --vqa"
        fi

        echo "Executing: $CMD"
        eval $CMD &
    done

    wait
    echo "Completed task: $task_name"
}

# run inference
for task_name in "${!tasks[@]}"; do
    run_task "$task_name" "${tasks[$task_name]}"
done

echo "All tasks completed!"

bash evaluate.sh --model "$model_name" --tag "$tag"