import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math
import sys

import re

import math

from modules.prompt import generate_unrelated_distraction_prompts, RandomImageIterator

from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import yaml


llava_visual_format = """{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:"""
qa_image_prompt = """You are an expert at question answering. Given the question and the context image of the question, please output the answer. Please provide a concise option, no explanation and further question."""

def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def load_dataset(data_name, mode, task):
    if mode in ['text_heavy', 'pure_text']:
        if "infoseek" in data_name:
            dataset_nickname = "infoseek"
            with open("data/infoseek/sampled_val_mc.json", "r") as fin:
                dataset = json.load(fin)
        elif "openbookqa" in data_name:
            dataset_nickname = "OpenBookQA"
            with open(f"data/OpenBookQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)
        elif "mmlu" in data_name:
            dataset_nickname = "MMLU"
            with open(f"data/MMLU/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

    elif mode == 'image_heavy':
        if "caltech-101" in data_name:
            dataset_nickname = "caltech-101"
            with open(f"data/caltech-101/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)
        if "mini-imagenet" in data_name:
            dataset_nickname = "mini-imagenet"
            with open(f"data/mini-imagenet/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

    elif mode == 'VQA':

        if "viquae" in data_name:
            dataset_nickname = "viquae"
            with open(f"data/VQADatasets/viquae/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "imagewikiqa" in data_name:
            dataset_nickname = "ImageWikiQA"
            with open(f"data/VQADatasets/ImageWikiQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if 'scienceqa' in data_name:
            dataset_nickname = "ScienceQA"
            with open(f"data/VQADatasets/ScienceQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "aokvqa" in data_name:
            dataset_nickname = "A-OKVQA"
            with open(f"data/VQADatasets/A-OKVQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "llava-instruct-665k" in data_name:
            dataset_nickname = "LLaVa-Instruct-665K"
            with open(f"/data/users/your name/data/LLaVa-Instruct-665K/llava_v1_5_mix665k_img.json", "r") as fin:
                dataset = json.load(fin)

        if "textcaps" in data_name:
            dataset_nickname = "TextCaps"
            with open(f"/data/users/your name/data/TextCaps/TextCaps_conversation.json", "r") as fin:
                dataset = json.load(fin)

        if "seed-bench-img" in data_name:
            dataset_nickname = "SEED-Bench-Img"
            with open(f"/data/users/your name/data/SEED-Bench-Img/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin)

        if "mmbench-en" in data_name:
            dataset_nickname = "MMBench-EN"
            with open(f"/data/users/your name/data/MMBench-EN/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin) 
    
    else:
        raise ValueError("Invalid mode")
    return dataset_nickname, dataset


def text_aug_for_ImageHeavy(batch, dataset_nickname):
    aug_contexts = defaultdict(list)
    
    for sample in ['related_text', 'origin', 'unrelated_text']:
        for data in batch:
            question = data["question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"

            image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))

            if sample == 'related_text':
                interfere_facts = list(data.get("facts").values())
                add_text = np.random.choice(interfere_facts)
                question = add_text + ' ' + question
            elif sample == 'origin':
                pass
            elif sample == 'unrelated_text':
                add_text = generate_unrelated_distraction_prompts()
                question = add_text + question
            elif sample == 'TextFooler':
                raise NotImplementedError
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            context = {"text": text, "image": image}
            aug_contexts[sample].append(context)
    return aug_contexts

def image_aug_for_TextHeavy(batch, dataset_nickname, model_nickname=None):
    aug_contexts = defaultdict(list)
    image_loader = RandomImageIterator(f"data/VQADatasets/MMVP/images/")
    for sample in ['random', 'switch', 'full_black', 'full_white']:
        for data in batch:
            question = data["question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            if sample == 'random':
                image = Image.fromarray(np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8))
            elif sample == 'switch':
                image_path = image_loader.get_random()
                image = Image.open(image_path)
            elif sample == 'full_black':
                image = Image.new('RGB', (336, 336), color = 'black')
            elif sample == 'full_white':
                image = Image.new('RGB', (336, 336), color = 'white')
            elif sample == "FGSM":
                base_dir = f"/data/users/your name/data/{dataset_nickname}/{model_nickname}/full_black/FGSM_Images_0.05"
                image = Image.open(os.path.join(base_dir, f"{data['id']}.png"))
            elif sample == "PGD":
                base_dir = f"/data/users/your name/data/{dataset_nickname}/{model_nickname}/full_black/PGD_Images_0.05"
                image = Image.open(os.path.join(base_dir, f"{data['id']}.png"))
            context = {"text": text, "image": image}
            aug_contexts[sample].append(context)
    return aug_contexts

def transform_image_path(data_nickname, image_path_or_text):
    if data_nickname in ["caltech-101", "mini-imagenet"]:
        return f"data/{data_nickname}/images/{image_path_or_text}"
    elif data_nickname in ["viquae", "ImageWikiQA", "ScienceQA"]:
        return os.path.join(f"data/VQADatasets/{data_nickname}/images/", image_path_or_text)
    elif data_nickname in ["A-OKVQA"]:
        return os.path.join("/data/users/your name/data/LLaVa-Instruct-665K/coco/images/", image_path_or_text)
    elif data_nickname in ['TextCaps', 'SEED-Bench-Img']:
        return os.path.join(f"/data/users/your name/data/{data_nickname}/", image_path_or_text)
    elif data_nickname in ["LLaVa-Instruct-665K"]:
        return os.path.join(f"/data/users/your name/data/LLaVa-Instruct-665K/", image_path_or_text)
    elif data_nickname in ["MMBench-EN"]:
        import base64
        from io import BytesIO
        return BytesIO(base64.b64decode(image_path_or_text))
    else:
        raise ValueError("Invalid dataset nickname")
    
def vqa_origin_format(batch, dataset_nickname):
    aug_contexts = defaultdict(list)
    for data in batch:
        question = data["question"]
        choices = data["multiple_choices"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"

        image_path = transform_image_path(dataset_nickname, data["image"])
        image = Image.open(image_path)

        text = f"Question:\n{question}\nOption:\n{choices_text}"
        context = {"text": text, "image" : image}
        aug_contexts['origin'].append(context)
    return aug_contexts

def write_prediction_into_file(output_path, final_choices):
    for data_id, final_prediction in final_choices.items():
        if final_prediction:
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: final_prediction})}\n")
                
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_final_choice(text, dataset_nickname):
    if dataset_nickname in ["caltech-101", "mini-imagenet"]:    
        pattern = r'([A-Za-z])'
    else:
        pattern = r'([A-Da-d])'
    matches = re.findall(pattern, text)
    # Find the first matched answer
    return matches[0].upper() if matches else None

def process_context(contexts): 
    batch_texts = [ctx["text"] for ctx in contexts]
    batch_images = [ctx["image"] for ctx in contexts]
    batch_format_texts = [llava_visual_format.format(system_prompt=qa_image_prompt, user_input=text) for text in batch_texts]
    
    return batch_format_texts, batch_images
    
def eval_model(args):
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
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llava-v1.5-13b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = 'left'
    # if image_processor is None:
    #     print("⚠️ image_processor is None, fallback to CLIPImageProcessor.")
    #     from transformers import CLIPImageProcessor
    #     image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
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
                
                batch_texts, batch_images = process_context(contexts)
                
                image_tensors = torch.stack([
                    image_processor.preprocess(im, return_tensors='pt')['pixel_values'][0] for im in batch_images
                ]).half().to(model.device)
                input_ids_list = [
                    tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for text in batch_texts
                ]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [ids.squeeze(0) for ids in input_ids_list], batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(model.device)
                # input_ids = tokenizer.pad(
                #     {"input_ids": [ids.squeeze(0) for ids in input_ids_list]},
                #     padding=True,
                #     return_tensors="pt"
                # )["input_ids"].to(model.device)
                
                conv = conv_templates[args.conv_mode].copy()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                    
                batch_outputs = tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )

                final_choices = {}
                for data_id, output in zip(batch_data_ids, batch_outputs):
                    output = output.strip()
                    if output.endswith(stop_str):
                        output = output[:-len(stop_str)]
                    output = output.strip()

                    choice = extract_final_choice(output, dataset_nickname)
                    
                    print(f"[Sample ID: {data_id}] Output: {repr(output)} --> Parsed choice: {choice}")
                    
                    final_choices[data_id] = choice
                    
                write_prediction_into_file(output_path, final_choices)

        if mode == "image_heavy" and selected_image_heavy_samples:
            batch_contexts = text_aug_for_ImageHeavy(batch, dataset_nickname)
            for sample in selected_image_heavy_samples:
                print(f"🔹 Testing Image-Heavy Sample: {sample}")
                contexts = batch_contexts[sample]
                output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{sample}_{tag_name}.txt")
                
                batch_texts, batch_images = process_context(contexts)
                image_tensors = torch.stack([
                    image_processor.preprocess(im, return_tensors='pt')['pixel_values'][0] for im in batch_images
                ]).half().to(model.device)
                input_ids_list = [
                    tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for text in batch_texts
                ]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [ids.squeeze(0) for ids in input_ids_list], batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(model.device)
                
                conv = conv_templates[args.conv_mode].copy()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                    
                batch_outputs = tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )

                final_choices = {}
                for data_id, output in zip(batch_data_ids, batch_outputs):
                    output = output.strip()
                    if output.endswith(stop_str):
                        output = output[:-len(stop_str)]
                    output = output.strip()

                    choice = extract_final_choice(output, dataset_nickname)
                    
                    print(f"[Sample ID: {data_id}] Output: {repr(output)} --> Parsed choice: {choice}")
                    
                    final_choices[data_id] = choice
                    
                write_prediction_into_file(output_path, final_choices)

        if mode == "VQA" and run_vqa:
            print(f"🔹 Testing VQA Sample")
            if dataset_nickname in ['SEED-Bench-Img', 'MMBench-EN', 'ScienceQA']:
                contexts = vqa_origin_format(batch, dataset_nickname)
                output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_origin_{tag_name}.txt")
                
                batch_texts, batch_images = process_context(contexts['origin'])
                image_tensors = torch.stack([
                    image_processor.preprocess(im, return_tensors='pt')['pixel_values'][0] for im in batch_images
                ]).half().to(model.device)
                input_ids_list = [
                    tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for text in batch_texts
                ]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [ids.squeeze(0) for ids in input_ids_list], batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(model.device)
                
                conv = conv_templates[args.conv_mode].copy()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                    
                batch_outputs = tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )

                final_choices = {}
                for data_id, output in zip(batch_data_ids, batch_outputs):
                    output = output.strip()
                    if output.endswith(stop_str):
                        output = output[:-len(stop_str)]
                    output = output.strip()

                    choice = extract_final_choice(output, dataset_nickname)
                    
                    print(f"[Sample ID: {data_id}] Output: {repr(output)} --> Parsed choice: {choice}")
                    
                    final_choices[data_id] = choice
                    
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
    parse.add_argument("--model-name", type=str, default="llava-1.5-7b", help="the model name")
    parse.add_argument("--model-base", type=str, default=None)
    parse.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parse.add_argument("--batch_size", type=int, default=8, help="the batch size")
    parse.add_argument("--tag", type=str, default="", help="the tag of the output file")
    parse.add_argument("--num-chunks", type=int, default=1)
    parse.add_argument("--chunk-idx", type=int, default=0)
    parse.add_argument("--all", action="store_true", help="Test all samples")
    parse.add_argument("--text_heavy", nargs="*", default=[], help="Select specific perturbations for text-heavy tasks")
    parse.add_argument("--image_heavy", nargs="*", default=[], help="Select specific perturbations for image-heavy tasks")
    parse.add_argument("--vqa", action="store_true", help="Enable VQA testing")
    parse.add_argument("--num_beams", type=int, default=1)
    parse.add_argument("--conv-mode", type=str, default="llava_v1")
    
    args = parse.parse_args()
    configs = read_yaml(path=args.config)
    configs.update(
        {
        "num_chunks": args.num_chunks,
        "chunk_idx": args.chunk_idx,
        "model_base": args.model_base,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "tag": args.tag,
        "all": args.all,  
        "text_heavy": args.text_heavy,  
        "image_heavy": args.image_heavy,  
        "vqa": args.vqa,  
        "num_beams": args.num_beams,
        "conv_mode": args.conv_mode,
        }  
    )
    args = SimpleNamespace(**configs)

    eval_model(args)        
