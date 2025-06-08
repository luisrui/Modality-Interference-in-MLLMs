import os
import json
import argparse
import sys
import math
import numpy as np

from tqdm import tqdm
from modules.utils.read_utils import read_yaml, load_dataset, get_chunk
from modules.utils.model_utils import model_predict_and_extract
from modules.utils.aug_utils import text_aug_for_ImageHeavy, image_aug_for_TextHeavy, vqa_origin_format
from modules.models import get_model_from_checkpoint
from modules.prompt import qa_image_prompt

import torchvision.transforms as T
from types import SimpleNamespace

def write_prediction_into_file(output_path, final_choices):
    # for data_id, final_choice in final_choices.items():
    #     if final_choice:
    #         with open(output_path, "a+") as fout:
    #             fout.write(f"{json.dumps({data_id: final_choice})}\n")
    for data_id, final_prediction in final_choices.items():
        if final_prediction:
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: final_prediction})}\n")

def predict(args):
    # load dataset
    if "text_heavy" in args.dataset: ## Text heavy task
        mode = "text_heavy"
    elif "image_heavy" in args.dataset: ## Image heavy task
        mode = "image_heavy"
    elif "VQA" in args.dataset: ## Pure text task
        mode = "VQA"
    else:
        raise ValueError("Invalid mode or dataset")
    
    text_heavy_samples = ['random', 'switch', 'full_black', 'full_white']
    image_heavy_samples = ['related_text', 'origin', 'unrelated_text']

    if args.all:
        selected_text_heavy_samples = text_heavy_samples
        selected_image_heavy_samples = image_heavy_samples
        run_vqa = True
    else:
        selected_text_heavy_samples = args.text_heavy if args.text_heavy else []
        selected_image_heavy_samples = args.image_heavy if args.image_heavy else []
        run_vqa = args.vqa

    dataset_nickname, dataset = load_dataset(args.dataset, mode, 'test')
    model_nickname = args.model_name.split("/")[-1]
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    tag_name = args.tag[1:] if args.tag.startswith("_") else args.tag   
    output_dir = os.path.join(args.output_dir, dataset_nickname, model_nickname, mode, tag_name)
    os.makedirs(output_dir, exist_ok=True)

    model = get_model_from_checkpoint(args)(args, mode, initial_prompt=qa_image_prompt)
    
    batch_data_ids = []
    batch_contexts = []
    pb = tqdm(total=len(dataset), desc="Processing samples")
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i + args.batch_size]
        batch_data_ids = [data["id"] for data in batch]
        if mode == "text_heavy" and selected_text_heavy_samples:
            batch_contexts = image_aug_for_TextHeavy(batch, dataset_nickname, model_nickname)
            for sample in selected_text_heavy_samples:
                print(f"🔹 Testing Text-Heavy Sample: {sample}")
                contexts = batch_contexts[sample]
                output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{sample}_{tag_name}.txt")
                final_choices = model_predict_and_extract(args, batch_data_ids, contexts, model, dataset_nickname)
                write_prediction_into_file(output_path, final_choices)

        if mode == "image_heavy" and selected_image_heavy_samples:
            batch_contexts = text_aug_for_ImageHeavy(batch, dataset_nickname)
            for sample in selected_image_heavy_samples:
                print(f"🔹 Testing Image-Heavy Sample: {sample}")
                contexts = batch_contexts[sample]
                output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{sample}_{tag_name}.txt")
                final_choices = model_predict_and_extract(args, batch_data_ids, contexts, model, dataset_nickname)
                write_prediction_into_file(output_path, final_choices)

        if mode == "VQA" and run_vqa:
            print(f"🔹 Testing VQA Sample")
            if dataset_nickname in ['SEED-Bench-Img', 'MMBench-EN', 'ScienceQA']:
                batch_contexts = vqa_origin_format(batch, dataset_nickname)
                output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_origin_{tag_name}.txt")
                final_choices = model_predict_and_extract(args, batch_data_ids, batch_contexts['origin'], model, dataset_nickname)
                write_prediction_into_file(output_path, final_choices)
            else:
                pass
                        
        pb.update(len(batch))

if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        config_path = sys.argv[1]
        sys.argv = [sys.argv[0], '--config', config_path]

    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    parse.add_argument("--model_name", type=str, default="llava-1.5-7b", help="the model name")
    parse.add_argument("--checkpoint_path", type=str, default="", help="the path of the checkpoint")
    parse.add_argument("--batch_size", type=int, default=8, help="the batch size")
    parse.add_argument("--tag", type=str, default="", help="the tag of the output file")
    parse.add_argument("--num-chunks", type=int, default=1)
    parse.add_argument("--chunk-idx", type=int, default=0)
    parse.add_argument("--all", action="store_true", help="Test all samples")
    parse.add_argument("--text_heavy", nargs="*", default=[], help="Select specific perturbations for text-heavy tasks")
    parse.add_argument("--image_heavy", nargs="*", default=[], help="Select specific perturbations for image-heavy tasks")
    parse.add_argument("--vqa", action="store_true", help="Enable VQA testing")
    
    args = parse.parse_args()
    configs = read_yaml(path=args.config)
    configs.update(
        {
        "num_chunks": args.num_chunks,
        "chunk_idx": args.chunk_idx,
        "model_name": args.model_name,
        "checkpoint_path": args.checkpoint_path,
        "batch_size": args.batch_size,
        "tag": args.tag,
        "all": args.all,  
        "text_heavy": args.text_heavy,  
        "image_heavy": args.image_heavy,  
        "vqa": args.vqa,  
        }  
    )
    args = SimpleNamespace(**configs)

    predict(args)        