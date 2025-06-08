import os
import json
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from modules.utils.read_utils import read_yaml, load_dataset, extract_final_choice
from modules.utils.model_utils import model_path
from modules.models import get_model
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
qa_blend_prompt, paraphrase, generate_unrelated_distraction_prompts, RandomImageIterator)

import torchvision.transforms as T
from types import SimpleNamespace

def read_pred_file(input):
    preds = {}
    with open(input, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    parse.add_argument("--input", type=str, default=None, help="the relative path of input file")
    config = parse.parse_args()
    args = read_yaml(path=config.config)
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
            output_path = os.path.join(output_dir, f"sub_{dataset_nickname}_{args.num_options}_{mode}_{args.sample}_new.txt")
        else:
            output_path = os.path.join(output_dir, f"{dataset_nickname}_{args.num_options}_{mode}_{args.sample}_new.txt")
    else:
        output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{args.sample}_new.txt")

    if mode == "pure_text":
        model = get_model(args)(args, mode, initial_prompt=qa_prompt)
    else:
        model = get_model(args)(args, mode, initial_prompt=qa_image_prompt)

    if args.sample == "random":
        template_picture = "data/VQADatasets/viquae/images/512px-'Endurance'_(1912)_in_the_ice_RMG_L7806.jpg"
    elif args.sample == "switch":
        image_loader = RandomImageIterator(f"data/VQADatasets/viquae/images/")
    pb = tqdm(range(len(dataset)))

    preds = read_pred_file(config.input)
    for data in dataset:
        data_id = data["id"]
        pred = preds.get(data_id)
        if pred not in ['A', 'B', 'C', 'D']:
            question = data["question"]
            choices = data["multiple_choices"]
            choices_text = ""
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            
            if mode == "text_heavy":
                ### Interference with image when doing text heavy task
                entity = data.get("entity")
                if entity:
                    question = f"This is a question about {entity}.\n{question}"
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                if args.sample == 'origin':
                    pass
                elif args.sample == 'random':
                    image = Image.open(template_picture)
                    width, height = image.size
                    random_pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                    image = Image.fromarray(random_pixels)
                elif args.sample == 'switch':
                    image_path = image_loader.get_random()
                    image = Image.open(image_path)
                elif args.sample == 'full_black':
                    image = Image.new('RGB', (336, 336), color = 'black')
                elif args.sample == 'full_white':
                    image = Image.new('RGB', (336, 336), color = 'white')
                context = {"text": text, "image": image}
            elif mode == "image_heavy":
                image = Image.open(os.path.join(f"data/{dataset_nickname}/test", data["image"]))
                if args.sample == 'origin':
                    pass
                elif args.sample == 'random':
                    question = paraphrase(question)
                elif args.sample == 'unrelated_text':
                    add_text = generate_unrelated_distraction_prompts()
                    question = add_text + question
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                context = {"text": text, "image": image}
            elif mode == "pure_text":
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                context = {"text": text}
            #context.update({"is_scored": args.is_scored})
            final_choice = "None"
            try:
                for i in range(args.retry_thres):
                    answer = model.chat(**context)
                    if extract_final_choice(answer, dataset_nickname):
                        final_choice = extract_final_choice(answer, dataset_nickname)
                        break
                    else:
                        final_choice = answer
                    context["text"] = context["text"] + f"\nPlease direcly answer with the option label!"
                with open(output_path, "a+") as fout:
                    fout.write(f"{json.dumps({data_id: final_choice})}\n")
            except Exception as e:
                print(f"Error in data_id: {data_id}, error: {e}")
        else:
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: pred})}\n")
        pb.update(1)

if __name__ == "__main__":
    main()            