from lmdeploy import pipeline, TurbomindEngineConfig
import os
from tqdm import tqdm
import json
import argparse
import torch
from functools import partial
from types import SimpleNamespace
from PIL import Image

from modules.utils.read_utils import read_yaml, load_dataset, extract_final_choice, load_image
from modules.utils.model_utils import model_path
from modules.models import get_model
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
qa_blend_prompt, paraphrase, generate_unrelated_distraction_prompts, RandomImageIterator)

def process_batch_inference(args, model_path, dataset, dataset_nickname, batch_size=4, mode="image_heavy", output_path=None):
    pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(
        session_len=4000,
        cache_max_entry_count=0.2,
        rope_scaling_factor=1.0,
        trust_remote_code=True
    ))

    if args.sample == "switch":
        image_loader = RandomImageIterator(f"data/VQADatasets/viquae/images/")

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_data = dataset[i:min(i + batch_size, len(dataset))]

        prompts = []
        data_ids = []
        
        for data in batch_data:
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
                    random_array = torch.rand(336, 336, 3).numpy() * 255
                    image = Image.fromarray(random_array.astype('uint8'))
                elif args.sample == 'switch': ## Inference MLLM with random image and text input
                    image_path = image_loader.get_random()
                    image = Image.open(image_path).convert('RGB')
                elif args.sample == 'full_black':
                    image = Image.fromarray(torch.zeros(336, 336, 3).numpy().astype('uint8'))
                elif args.sample == 'full_white':
                    image = Image.fromarray(torch.ones(336, 336, 3).numpy().astype('uint8') * 255)
                text = qa_image_prompt + "USER: <image>\n" + text
                prompts.append((text, image))
            if mode == "image_heavy":
                ### Interference with text when doing image heavy task
                image_path = os.path.join(f"data/{dataset_nickname}/images", data["image"])
                image = Image.open(image_path).convert('RGB')
                if args.sample == 'origin':
                    pass
                elif args.sample == 'paraphrase':
                    question = paraphrase()
                elif args.sample == 'unrelated_text':
                    add_text = generate_unrelated_distraction_prompts()
                    question = add_text + question
                elif args.sample == 'related_text':
                    add_text = paraphrase()
                    question = add_text + question
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                text = qa_image_prompt + "USER: <image>\n" + text
                prompts.append((text, image))
            elif mode == "pure_text":
                text = f"Question:\n{question}\nOption:\n{choices_text}"
                entity = data.get('entity')
                if entity is None:
                    caption = ""
                else:
                    caption = f"This is a question about {entity}."
                text = qa_prompt + "USER: " + caption + "\n" + text
                prompts.append(text)
            
            data_ids.append(data_id)

        responses = pipe(prompts)

        for idx, data_tuple in enumerate(zip(data_ids, responses)):
            data_id, response = data_tuple
            final_choice = extract_final_choice(response.text) or response.text

            if output_path:
                with open(output_path, "a+") as fout:
                    fout.write(f"{json.dumps({data_id: final_choice})}\n")


def optimized_main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    args = parse.parse_args()
    args = read_yaml(path=args.config)
    args = SimpleNamespace(**args)
    
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
    
    if model_nickname in model_path.keys():
        path = model_path[model_nickname]
    else:
        print('error model name')
        return 
    
    process_batch_inference(
        args=args,
        model_path=path,
        dataset=dataset,
        dataset_nickname=dataset_nickname,
        batch_size=args.batch_size,
        mode=mode,
        output_path=output_path
    )

if __name__ == "__main__":
    optimized_main()