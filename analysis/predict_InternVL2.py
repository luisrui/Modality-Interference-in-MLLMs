import os
import json
import argparse
import numpy as np
import torch 

from types import SimpleNamespace
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from modules.utils.read_utils import read_yaml, load_dataset, extract_final_choice, load_image
from modules.utils.model_utils import model_path
from modules.models import get_model
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
qa_blend_prompt, paraphrase, generate_unrelated_distraction_prompts,RandomImageIterator)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    args = parse.parse_args()
    args = read_yaml(path=args.config)
    args = SimpleNamespace(**args)

    # load dataset
    if "text_heavy" in args.dataset: ## Text heavy task
        mode = "text_heavy"
    elif "image_heavy" in args.dataset: ## Image heavy task
        mode = "image_heavy"
    elif "pure_text" in args.dataset: ## Pure text task
        mode = "pure_text"
    dataset_nickname, dataset = load_dataset(args, mode)
    model_nickname = args.model_name.split("/")[-1]

    output_dir = os.path.join(args.output_dir, dataset_nickname, model_nickname, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if mode == "image_heavy":
        if args.is_sub:
            output_path = os.path.join(output_dir, f"sub_{dataset_nickname}_{args.num_options}_{mode}_{args.sample}.txt")
        else:
            output_path = os.path.join(output_dir, f"{dataset_nickname}_{args.num_options}_{mode}_{args.sample}.txt")
    else:
        output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{args.sample}.txt")
    
    # if args.is_scored:
    #     output_path += ".score"
    if model_nickname in model_path.keys():
        path = model_path[model_nickname]
    else:
        print('error model name')
        return 
    
    if mode == "pure_text":
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    else:
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=True)

    if args.sample == "switch":
        image_loader = RandomImageIterator(f"data/VQADatasets/viquae/images/")
    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        question = data["question"]
        choices = data["multiple_choices"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"
        
        if mode == "text_heavy":
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            ### Interference with image when doing text heavy task
            if args.sample == 'origin': ## Inference MLLM with only text input
                pass
            elif args.sample == 'random': ## Inference MLLM with random pixels and text input
                image = torch.rand((1, 3, 448, 448)).to(torch.bfloat16).cuda() * 255
            elif args.sample == 'switch': ## Inference MLLM with random image and text input
                image_path = image_loader.get_random()
                image = load_image(image_path).to(torch.bfloat16).cuda()
            elif args.sample == 'full_black':
                image = torch.zeros((1, 3, 448, 448)).to(torch.bfloat16).cuda()
            elif args.sample == 'full_white':
                image = torch.ones((1, 3, 448, 448)).to(torch.bfloat16).cuda() 
            text = qa_image_prompt + "USER: <image>\n" + text
            context = {"text": text, "image": image}
        if mode == "image_heavy":
            ### Interference with text when doing image heavy task
            try:
                image = load_image(os.path.join(f"data/{dataset_nickname}/images", data["image"])).to(torch.bfloat16).cuda()
            except:
                image = load_image(os.path.join(f"data/{dataset_nickname}/test", data["image"])).to(torch.bfloat16).cuda()
            if args.sample == 'origin':
                pass
            elif args.sample == 'paraphrase':
                question = paraphrase()
            elif args.sample == 'unrelated_text':
                add_text = generate_unrelated_distraction_prompts()
                question = add_text + question
            elif args.sample == 'related_text':
                interfere_facts = list(data.get("facts").values())
                add_text = np.random.choice(interfere_facts)
                question = add_text + ' ' + question
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            text = qa_image_prompt + "USER: <image>\n" + text
            context = {"text": text, "image": image}
        elif mode == "pure_text":
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            entity = data.get('entity')
            if entity is None:
                caption = ""
            else:
                caption = f"This is a question about {entity}."
            text = qa_prompt + "USER: " + caption + "\n" + text
            #text = qa_prompt + "USER: " + text
            context = {"text": text}
        final_choice = "None"
        for i in range(args.retry_thres):
            if mode == "pure_text":
                answer, history = model.chat(tokenizer, context['text'], history=[])
            else:
                answer = model.chat(tokenizer, context['image'], context['text'], generation_config) # Get the answer through model inference
            if extract_final_choice(answer, dataset_nickname):
                final_choice = extract_final_choice(answer, dataset_nickname)
                break
            final_choice = answer
            context["text"] = context["text"] + f"\nPlease direcly choose the option!"
        with open(output_path, "a+") as fout:
            fout.write(f"{json.dumps({data_id: final_choice})}\n")
        pb.update(1)

if __name__ == "__main__":
    main()            