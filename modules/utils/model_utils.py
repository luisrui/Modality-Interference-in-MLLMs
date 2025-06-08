import json
import torch
import os

from transformers import LlavaProcessor, InstructBlipProcessor, AutoProcessor
from transformers import LlavaForConditionalGeneration, InstructBlipForConditionalGeneration
try:
    from transformers import  Qwen2_5_VLForConditionalGeneration
except: 
    pass


from .read_utils import extract_final_choice
from ..models import * 

model_path = {
    'InternLM2-1.8b' : 'internlm/internlm2-chat-1_8b',
    'InternVL2-2B' : 'OpenGVLab/InternVL2-2B',
    'InternVL2-4B' : 'OpenGVLab/InternVL2-4B',
    'InternVL2-8B' : 'OpenGVLab/InternVL2-8B',
    'internlm2-chat-1_8b' : 'internlm/internlm2-chat-1_8b',
    'vicuna-7b' : 'lmsys/vicuna-7b-v1.5'
}

def get_model_and_processor(args):
    consistency_type = args.consistency_type if hasattr(args, 'consistency_type') else ""
    adversarial_type = args.adversarial_type if hasattr(args, 'adversarial_type') else ""
    if 'llava-1.5' in args.model:
        if '7b' in args.model:
            checkpoint_path = "llava-hf/llava-1.5-7b-hf" if not args.resume_from_checkpoint else args.resume_from_checkpoint
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        elif '13b' in args.model:
            checkpoint_path = "llava-hf/llava-1.5-13b-hf" if not args.resume_from_checkpoint else args.resume_from_checkpoint
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
        if consistency_type and adversarial_type:
            if 'KL' in consistency_type and 'PGD' in adversarial_type:
                model = LLaVAForKLConsistencyPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                ) 
            elif 'JS' in consistency_type and 'PGD' in adversarial_type:
                model = LLaVAForJSdivergenceAndPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
            elif 'MSE' in consistency_type and 'PGD' in adversarial_type:
                model = LLaVAForMSEdivergenceAndPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        elif consistency_type: # Only use consistency loss and consistency regularization
            if 'JS' in consistency_type:
                model = LLaVAForJSdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
            elif 'KL' in consistency_type:
                model = LLaVAForKLdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
            elif 'MSE' in consistency_type:
                model = LLaVAForMSEdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        elif adversarial_type: # Only use adversarial attack and perturbation reguralzation
            if "RG" in adversarial_type:
                model = LLaVAForRGPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
            elif "PGD" in adversarial_type:
                model = LLaVAForPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        else: # using nothing
            model = LlavaForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype = torch.float16,
            )
            
    elif 'instructblip-vicuna-7b' in args.model:
        checkpoint_path = "Salesforce/instructblip-vicuna-7b" if not args.resume_from_checkpoint else args.resume_from_checkpoint
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        if consistency_type and adversarial_type: # Use both augmentations
            if 'KL' in consistency_type and 'PGD' in adversarial_type:
                model = InstructBlipForKLConsistencyPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        elif consistency_type: # Only use consistency loss and consistency regularization
            if 'KL' in args.consistency_type:
                model = InstructBlipForKLdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
            if "JS" in args.consistency_type:
                model = InstructBlipForJSdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        elif adversarial_type: # Only use adversarial attack and perturbation reguralzation
            if "PGD" in args.adversarial_type:
                model = InstructBlipForPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = torch.float16,
                )
        else:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype = torch.float16,
            )

    elif 'qwen2.5-vl' in args.model:
        if "7b" in args.model:
            checkpoint_path = "Qwen/Qwen2.5-VL-7B-Instruct" if not args.resume_from_checkpoint else args.resume_from_checkpoint
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            use_type = torch.float16
        elif "3b" in args.model:
            checkpoint_path = "Qwen/Qwen2.5-VL-3B-Instruct" if not args.resume_from_checkpoint else args.resume_from_checkpoint
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            use_type = torch.bfloat16
        if consistency_type and adversarial_type:
            if 'KL' in consistency_type and 'PGD' in adversarial_type:
                model = Qwen2_5_VLForKLConsistencyPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type
                )
            elif 'JS' in consistency_type and 'PGD' in adversarial_type:
                model = Qwen2_5_VLForJSdivergenceAndPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type,
                )
        elif consistency_type: # Only use consistency loss and consistency regularization
            if 'KL' in consistency_type:
                model = Qwen2_5_VLForKLdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type,
                )
            elif 'JS' in consistency_type:
                model = Qwen2_5_VLForJSdivergence.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type,
                )
        elif adversarial_type: # Only use adversarial attack and perturbation reguralzation
            if "RG" in adversarial_type:
                model = Qwen2_5_VLForRGPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type,
                )
            elif "PGD" in adversarial_type:
                model = Qwen2_5_VLForPGDPerturbation.from_pretrained(
                        checkpoint_path,
                        torch_dtype = use_type,
                )
        else: # using nothing
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype = use_type,
            )
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    return model, processor

def get_trainer(args, model, training_args, train_data, test_data, metrics, data_collator, train_loader=None):
    consistency_type = args.consistency_type if hasattr(args, 'consistency_type') else ""
    adversarial_type = args.adversarial_type if hasattr(args, 'adversarial_type') else ""
    if consistency_type:
        if "PGD" in adversarial_type:
            return ConsistencyPGDPerturbationTrainer(
                alpha=args.alpha,
                epsilon=float(args.epsilon),
                pgd_steps=args.pgd_steps,
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=metrics, # set to f1 score or accuracy if necessary
                data_collator=data_collator,
                train_loader=train_loader
            )
        else: # No adversarial perturbation
            return ConsistencyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=metrics, # set to f1 score or accuracy if necessary
                data_collator=data_collator,
                train_loader=train_loader
            )
    else: # No Consistency Regularization 
        if "RG" in adversarial_type:
            return RGPerturbationTrainer(
                sigma=args.sigma,
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=metrics, # set to f1 score or accuracy if necessary
                data_collator=data_collator,
                train_loader=train_loader
            )
        if "PGD" in adversarial_type:
            return PGDPerturbationTrainer(
                alpha=args.alpha,
                epsilon=float(args.epsilon),
                pgd_steps=args.pgd_steps,
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=metrics, # set to f1 score or accuracy if necessary
                data_collator=data_collator,
                train_loader=train_loader
            )
        else:
            return BatchSampleTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=metrics, # set to f1 score or accuracy if necessary
                data_collator=data_collator,
                train_loader=train_loader
            )

def freeze_parameters(*modules, partial_freeze=None, exclude_names=None):
    exclude_names = exclude_names or []

    for module in modules:
        for name, param in module.named_parameters():
            if any(ex_key in name for ex_key in exclude_names):
                continue
            param.requires_grad = False

    if partial_freeze is not None:
        module, ratio = partial_freeze
        params = list(module.named_parameters())
        freeze_count = int(len(params) * ratio)
        for i, (name, param) in enumerate(params):
            if any(ex_key in name for ex_key in exclude_names):
                continue
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True

def model_predict_and_extract(args:dict, data_ids:str, contexts:dict, model, dataset_nickname:str) -> dict:
    results = {}
    batch_results = {}
    batch_ids = data_ids # batch_size of data_ids
    batch_contexts = contexts # batch_size of contexts
    for retry in range(args.retry_thres):
        try:
            batch_answers = model.chat(batch_contexts)

            for data_id, answer, context in zip(batch_ids, batch_answers, batch_contexts):
                if data_id not in batch_results:  # If not already processed
                    final_choice = extract_final_choice(answer, dataset_nickname)
                    if final_choice:
                        batch_results[data_id] = final_choice
                        results[data_id] = {'choice' : final_choice, 'answer' : answer}
                    elif retry == args.retry_thres - 1:  # Last Choice
                        batch_results[data_id] = answer
                        results[data_id] = {'choice' : final_choice, 'answer' : answer}

            remaining_contexts = []
            remaining_ids = []
            for data_id, context in zip(batch_ids, batch_contexts):
                if data_id not in batch_results:
                    context["text"] += "\nPlease directly choose the option with option number. No further explanation."
                    remaining_contexts.append(context)
                    remaining_ids.append(data_id)

            if len(remaining_contexts) == 0:
                break
                
            batch_contexts = remaining_contexts
            batch_ids = remaining_ids
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            for data_id in batch_ids:
                if data_id not in results:
                    print(f"Error in data_id: {data_id}, error: {e}")
                    results[data_id] = None
            break
    
    return results

def model_predict_and_extract_multi_img(args, data_id, context, model, dataset_nickname):
    '''
    Make model prediction with one answer including multi images for each inference
    '''

    for retry in range(args.retry_thres):
        try:
            answer = model.predict(context)
            
            final_choice = extract_final_choice(answer, dataset_nickname)
            if extract_final_choice(answer, dataset_nickname):
                final_choice = extract_final_choice(answer, dataset_nickname)
                break
            else:
                final_choice = answer
            context["text"] = context["text"] + f"\nPlease directly choose the option with option number. No further explanation."

        except Exception as e:
            print(f"Error in data_id: {data_id}, error: {e}")
            break

    return final_choice

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
