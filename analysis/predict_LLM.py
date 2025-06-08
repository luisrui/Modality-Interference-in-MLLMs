import os
import json
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from modules.utils.read_utils import read_yaml, load_dataset, extract_final_choice
from modules.utils.model_utils import model_path
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
qa_blend_prompt, paraphrase, generate_unrelated_distraction_prompts, RandomImageIterator)

from types import SimpleNamespace

text_format = """"{system_prompt}\nUSER: {user_input}\nASSISTANT:"""

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/llava-v1.6-vicuna-7b/vicuna-7b/args.yaml", help="the relative path of argments file")
    args = parse.parse_args()
    args = read_yaml(path=args.config)
    args = SimpleNamespace(**args)

    model_nickname = args.model_name.split("/")[-1]
    
    # load dataset
    mode = "pure_text"
    dataset_nickname, dataset = load_dataset(args, mode)
    model_nickname = args.model_name.split("/")[-1]
    
    output_dir = os.path.join(args.output_dir, dataset_nickname, model_nickname, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{dataset_nickname}_{mode}_{args.sample}.txt")
    
    # if args.is_scored:
    #     output_path += ".score"
    if model_nickname in model_path.keys():
        path = model_path[model_nickname]
    else:
        print('error model name')
        return 
    
    model = LLM(
        model=path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    pb = tqdm(range(len(dataset)))
    for data in dataset:
        data_id = data["id"]
        question = data["question"]
        choices = data["multiple_choices"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"

        text = f"Question:\n{question}\nOption:\n{choices_text}"
        entity = data.get('entity')
        if entity is None:
            caption = ""
        else:
            caption = f"This is an question of {entity}."
        text = caption + "\n" + text
        text = text_format.format(system_prompt=qa_prompt, user_input=text)
        
        final_choice = "None"
        for i in range(args.retry_thres):
            answer = model.generate(
                text,
                sampling_params=sampling_params
            )
            answer_text =  answer[0].outputs[0].text
            # if extract_final_choice(answer_text):
            #     final_choice = extract_final_choice(answer_text)
            #     break

        if args.is_scored:
            target_tokens = ["A", "B", "C", "D"]
            target_scores = [-np.inf, -np.inf, -np.inf, -np.inf]
            logprobs = answer[0].outputs[0].logprobs[0]
            for _, logprob in logprobs.items():
                decoded_token = logprob.decoded_token.strip()
                try:
                    target_index = target_tokens.index(decoded_token)
                    target_scores[target_index] = logprob.logprob
                except:
                    continue
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: [answer[0].outputs[0].text, target_scores]})}\n")
        else:
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id:answer_text})}\n")
        pb.update(1)

if __name__ == "__main__":
    main()            