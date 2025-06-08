import json
import os
import argparse
import sys
import numpy as np
from tqdm import tqdm

from modules.utils import read_yaml, load_dataset, transform_image_path
from modules.prompt import generate_unrelated_distraction_prompts, RandomImageIterator

from types import SimpleNamespace
'''
Construct LLaVa-formatted data for supervised fine-tuning.
'''

def LLaVA_format_transfer_text_heavy(data:list, ratio:float = 1.0):

    formated_data = []

    sampled_data = np.random.choice(data, int(len(data) * ratio), replace=False)
    for qa_pair in tqdm(sampled_data):
        data_id = qa_pair["id"]
        question = qa_pair["question"]
        choices = qa_pair["multiple_choices"]
        choices_text = ""
        answer = qa_pair["multiple_choices_answer"]
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"
        text = f"Question:\n{question}\nOption:\n{choices_text}"
        #text = f"Question:\n{question}"
        #output = str(answer) + ': ' + str(choices[str(answer)]) + '.'
        output = str(answer)

        formated_data.append({
            "id" : data_id,
            'type' : 'text_heavy',
            "question" : text,
            "answer" : output
            }) # No need to add image path here
        
    if len(formated_data) != 0:
        print(f"Sample text heavy format data: {formated_data[0]}")
    return formated_data

def LLaVA_format_transfer_image_heavy(data:list, dataset_nickname:str, ratio:float = 1.0):
    formated_data = []

    sampled_data = np.random.choice(data, int(len(data) * ratio), replace=False)

    for qa_pair in tqdm(sampled_data):
        data_id = qa_pair["id"]
        question = qa_pair["question"]
        image_path = os.path.join(f"data/{dataset_nickname}/images", qa_pair["image"])
        choices = qa_pair["multiple_choices"]
        choices_text = ""
        answer = qa_pair["multiple_choices_answer"]
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"
        text_origin = f"Question:\n{question}\nOption:\n{choices_text}"
        #text_origin = f"Question:\n{question}"

        # text_unrelated_text = f"Question:\n{generate_unrelated_distraction_prompts() + question}\nOption:\n{choices_text}"
        
        interfere_facts = qa_pair.get("facts", [])
        # if len(interfere_facts) > 0:
        #     text_related_text = f"Question:\n{list(interfere_facts.values())[0] + question}\nOption:\n{choices_text}"
        # else:
        #     random_option = np.random.choice(list(choices.values()))
        #     text_related_text = f"Question:\n This picture seems to depict a {random_option}. {question}\nOption:\n{choices_text}"
        #text_unrelated_text = f"Question:\n{generate_unrelated_distraction_prompts() + question}"
        
        # interfere_facts = qa_pair.get("facts")
        # if len(interfere_facts) > 0:
        #     text_related_text = f"Question:\n{list(interfere_facts.values())[0] + question}"
        # else:
        #     random_option = np.random.choice(list(choices.values()))
        #     text_related_text = f"Question:\n This picture seems to depict a {random_option}. {question}"

        #output = str(answer) + ': ' + str(choices[str(answer)]) + '.'
        output = str(answer) 
        
        formated_data.append({
            "id" : data_id,
            'type' : 'image_heavy',
            "image": image_path, 
            "question" : text_origin,
            "answer" : output,
            "facts" : interfere_facts
        })
                
    if len(formated_data) != 0:
        print(f"Sample image heavy format data: {formated_data[0]}")
    return formated_data

def LLaVa_format_transfer_VQA(data, dataset_nickname, ratio:float = 1.0):  
    formated_data = []
    sampled_data = np.random.choice(data, int(len(data) * ratio), replace=False)
    if 'LLaVa-Instruct-665K' or "TextCaps" in dataset_nickname: 
        for conv in tqdm(sampled_data):
            data_id = conv["id"]
            image_path = transform_image_path(dataset_nickname, conv["image"])
            #os.path.join(f"data/VQADatasets/{dataset_nickname}/images", conv["image"])
            conversations = conv.get("conversations", None)
            if conversations:
                question = conversations[0].get('value')
                answer = conversations[1].get('value')
                formated_data.append({
                    "id" : data_id,
                    'type' : 'VQA',
                    "image" : image_path, 
                    "question" : question.replace("\n<image>", "").replace("<image>\n", "").replace("<image>", ""),
                    "answer" : answer
                    })
    else:    
        for qa_pair in tqdm(sampled_data):
            data_id = qa_pair["id"]
            question = qa_pair["question"]
            image_path = transform_image_path(dataset_nickname, qa_pair["image"])
            #os.path.join(f"data/VQADatasets/{dataset_nickname}/images", qa_pair["image"])
            choices = qa_pair["multiple_choices"]
            choices_text = ""
            answer = qa_pair["multiple_choices_answer"]
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"
            text = f"Question:\n{question}\nOption:\n{choices_text}"
            #text = f"Question:\n{question}"
            output = str(choices[str(answer)])
            #output = str(answer) + ': ' + str(choices[str(answer)]) + '.'

            formated_data.append({
                "id" : data_id,
                'type' : 'VQA',
                "image" : image_path, 
                "question" : text,
                "answer" : output
                })
    if len(formated_data) != 0:
        print(f"Sample multimodal format data: {formated_data[0]}")
    return formated_data

if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
    config_path = sys.argv[1]
    sys.argv = [sys.argv[0], '--config', config_path]

parse = argparse.ArgumentParser()
parse.add_argument("--config", type=str, default="./configs/data_split/default.yaml", help="the relative path of argments file")
args = parse.parse_args()
args = read_yaml(path=args.config)
args = SimpleNamespace(**args)

Image_datasets = args.datasets['image_heavy_datasets']
Image_ratios = args.datasets['image_heavy_data_ratio']

Text_datasets = args.datasets['text_heavy_datasets']
Text_ratios = args.datasets['text_heavy_data_ratio']

MM_datasets = args.datasets['multimodal_datasets']
MM_ratios = args.datasets['multimodal_data_ratio']

image_heavy_data = []
for data_name, data_ratio in zip(Image_datasets, Image_ratios):
    print('loading dataset:', data_name)
    dataset_nickname, dataset = load_dataset(data_name, "image_heavy", 'train')
    format_data = LLaVA_format_transfer_image_heavy(dataset, dataset_nickname, data_ratio)
    image_heavy_data.extend(format_data)

text_heavy_data = []
for data_name, data_ratio in zip(Text_datasets, Text_ratios):
    print('loading dataset:', data_name)
    dataset_nickname, dataset = load_dataset(data_name, "text_heavy", 'train')
    format_data = LLaVA_format_transfer_text_heavy(dataset, data_ratio)
    text_heavy_data.extend(format_data)

MM_data = []
for data_name, data_ratio in zip(MM_datasets, MM_ratios):
    print('loading dataset:', data_name)
    dataset_nickname, dataset = load_dataset(data_name, "VQA", 'train')
    format_data = LLaVa_format_transfer_VQA(dataset, dataset_nickname, data_ratio)
    MM_data.extend(format_data)

print(f"image_heavy_data: {len(image_heavy_data)}, text_heavy_data: {len(text_heavy_data)}, MM_data: {len(MM_data)}")

with open(f'data/{args.output_file}.json', 'w', encoding='utf-8') as f:
    json.dump(image_heavy_data + text_heavy_data + MM_data, f, indent=4, ensure_ascii=False)
print(f'successfully saved trainset data for llava finetuning, total qa pairs are {len(image_heavy_data) + len(text_heavy_data) + len(MM_data)}')



