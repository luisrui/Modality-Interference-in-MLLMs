import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import json
import sys
import math

'''
This script generates facts about the wrong options for image-heavy task interference in the multiple choice questions.
'''
def process_batch(batch_data, model, processor, device, text_format: str):
    batch_results = []
    
    all_options_to_process = []
    sample_indices = []
    
    for idx, qa_pair in enumerate(batch_data):
        correct_answer = qa_pair['multiple_choices'][qa_pair['multiple_choices_answer']]
        wrong_options = [opt for key, opt in qa_pair['multiple_choices'].items()
                        if qa_pair['multiple_choices'][key] != correct_answer]
        
        selected_wrong_options = random.sample(wrong_options, 1)
            
        for option in selected_wrong_options:
            all_options_to_process.append(option)
            sample_indices.append(idx)

    if all_options_to_process:
        texts = [
            text_format.format(
                system_prompt="Give me some fact about a certain concept. Please provide only Two sentences.",
                user_input=f'Example: Input: Tell me two facts about platypus. Output: The platypus is a unique animal that is found in the wild in Australia. It is a mammal, but it lays eggs like a bird.\nNow Tell me some facts about {option}.'
            )
            for option in all_options_to_process
        ]
        
        inputs = processor(
            text=texts,
            padding="longest",
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=processor.eos_token_id
            )
        
        generated_texts = []
        for idx, output in enumerate(outputs):
            decoded = processor.batch_decode(
                output[None, :],  
                skip_special_tokens=True
            )[0].strip()

            answer = decoded.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in decoded else decoded
            generated_texts.append(answer)
            print('The answer is:', answer)
        
        final_texts = [
            f"The picture seems to describe a {option}. {text}"
            for option, text in zip(all_options_to_process, generated_texts)
        ]

    for qa_pair in batch_data:
        qa_pair['facts'] = {}
    
    for idx, option, final_text in zip(sample_indices, all_options_to_process, final_texts):
        batch_data[idx]['facts'][option] = final_text
    
    return batch_data

def process_dataset(data, batch_size: int, model, processor, device, text_format: str):
    num_batches = math.ceil(len(data) / batch_size)
    processed_data = []
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        processed_batch = process_batch(batch_data, model, processor, device, text_format)
        processed_data.extend(processed_batch)
    
    return processed_data

def main():
    dataset = 'caltech-101'
    text_format = """{system_prompt}\nUSER: {user_input}\nASSISTANT:"""
    batch_size = 64

    data = json.load(open(f'./data/{dataset}/multiple_choice_data_train.json', 'r', encoding='utf-8'))

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    processor = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    model = AutoModelForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    processed_data = process_dataset(data, batch_size, model, processor, device, text_format)

    json.dump(
        processed_data,
        open(f'./data/{dataset}/multiple_choice_data_train_with_facts.json', 
             'w', 
             encoding='utf-8'),
        indent=4
    )

if __name__ == "__main__":
    main()
