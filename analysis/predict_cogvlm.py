import os
import json
import argparse
import sys
import numpy as np

from PIL import Image
from tqdm import tqdm
from modules.pretrained_models.local import get_model
from modules.pretrained_models.utils import read_yaml, load_dataset, extract_final_choice, transform_image_path
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
generate_unrelated_distraction_prompts, RandomImageIterator)

import torchvision.transforms as T
from types import SimpleNamespace

def main():
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        config_path = sys.argv[1]
        sys.argv = [sys.argv[0], '--config', config_path]

    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    parse.add_argument("--model_name", type=str, default="llava-1.5-7b", help="the model name")
    parse.add_argument("--tag", type=str, default="", help="the tag of the output file")
    
    args = parse.parse_args()
    configs = read_yaml(path=args.config)
    configs.update({
        "model_name": args.model_name,
        "tag": args.tag,
    }) 
    args = SimpleNamespace(**configs)

   # load dataset
    if "text_heavy" in args.dataset: ## Text heavy task
        mode = "text_heavy"
    elif "image_heavy" in args.dataset: ## Image heavy task
        mode = "image_heavy"
    elif "VQA" in args.dataset: ## Pure text task
        mode = "VQA"
    else:
        raise ValueError("Invalid mode")
    
    dataset_nickname, dataset = load_dataset(args.dataset, mode, 'test')
    model_nickname = args.model_name.split("/")[-1]

    output_dir = os.path.join(args.output_dir, dataset_nickname, model_nickname, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if mode == "pure_text":
        model = get_model(args)(args, mode, initial_prompt=qa_prompt)
    else:
        model = get_model(args)(args, mode, initial_prompt=qa_image_prompt)

    tag_name = args.tag[1:] if args.tag.startswith("_") else args.tag 

    image_loader = RandomImageIterator(f"data/VQADatasets/MMVP/images/")

    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        question = data["question"]
        choices = data["multiple_choices"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"

        if mode == "text_heavy":
            for sample in ['random', 'switch', 'full_black', 'full_white']:
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
                context = {"text": text, "image": image}

                try:
                    output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{sample}_{tag_name}.txt")
                    retry_times = 2
                    for i in range(retry_times):
                        answer = model.cogvlm_predict(context)
                        print(answer)
                        if extract_final_choice(answer, dataset_nickname):
                            final_choice = extract_final_choice(answer, dataset_nickname)
                            break
                        else:
                            final_choice = answer
                        context["text"] = context["text"] + f"\nPlease direcly choose the option with option number. No further explanation."
                    with open(output_path, "a+") as fout:
                        fout.write(f"{json.dumps({data_id: final_choice})}\n")
                except Exception as e:
                    print(f"Error in data_id: {data_id}, error: {e}")

        elif mode == "image_heavy":
            for sample in ['related_text', 'origin', 'unrelated_text']:
                image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
                if sample == "origin":
                    pass
                elif sample == 'related_text':
                    interfere_facts = list(data.get("facts").values())
                    add_text = np.random.choice(interfere_facts)
                    question = add_text + ' ' + question
                elif sample == 'unrelated_text':
                    add_text = generate_unrelated_distraction_prompts()
                    question = add_text + question
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                context = {"text": text, "image": image}

                try:
                    output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{sample}_{tag_name}.txt")
                    retry_times = 2
                    for i in range(retry_times):
                        answer = model.cogvlm_predict(context)
                        print(answer)
                        if extract_final_choice(answer, dataset_nickname):
                            final_choice = extract_final_choice(answer, dataset_nickname)
                            break
                        else:
                            final_choice = answer
                        context["text"] = context["text"] + f"\nPlease direcly choose the option with option number. No further explanation."
                    with open(output_path, "a+") as fout:
                        fout.write(f"{json.dumps({data_id: final_choice})}\n")
                except Exception as e:
                    print(f"Error in data_id: {data_id}, error: {e}")

        elif mode == "VQA":
            image_path = transform_image_path(dataset_nickname, data["image"])
            image = Image.open(image_path).convert("RGB")

            text = f"Question:\n{question}\nOption:\n{choices_text}"
            context = {"text": text, "image" : image}

            # try:
            #     output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{tag_name}.txt")
            #     retry_times = 2
            #     for i in range(retry_times):
            #         answer = model.cogvlm_predict(context)
            #         print(answer)
            #         if extract_final_choice(answer, dataset_nickname):
            #             final_choice = extract_final_choice(answer, dataset_nickname)
            #             break
            #         else:
            #             final_choice = answer
            #         context["text"] = context["text"] + f"\nPlease direcly choose the option with option number. No further explanation."
            #     with open(output_path, "a+") as fout:
            #         fout.write(f"{json.dumps({data_id: final_choice})}\n")
            # except Exception as e:
            #     print(f"Error in data_id: {data_id}, error: {e}")
            output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{tag_name}.txt")
            retry_times = 2
            for i in range(retry_times):
                answer = model.cogvlm_predict(context)
                print(answer)
                if extract_final_choice(answer, dataset_nickname):
                    final_choice = extract_final_choice(answer, dataset_nickname)
                    break
                else:
                    final_choice = answer
                context["text"] = context["text"] + f"\nPlease direcly choose the option with option number. No further explanation."
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: final_choice})}\n")

        pb.update(1)

if __name__ == "__main__":
    main()            