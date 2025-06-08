import os
import torch
import random
import numpy as np
import pandas as pd

from collections import defaultdict
from types import SimpleNamespace
from torchvision import transforms
from PIL import Image
from .read_utils import transform_image_path
from modules.prompt import (qa_prompt, qa_context_prompt, qa_image_prompt, 
qa_blend_prompt, paraphrase, generate_unrelated_distraction_prompts, RandomImageIterator, blip_example_format, llava_example_format)


image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]

def text_aug_for_ImageHeavy(batch, dataset_nickname):
    aug_contexts = defaultdict(list)
    gcg_adv_data = pd.read_json(f"data/{dataset_nickname}/perturb_GCG_K5L16.jsonl", lines=True)
    id_gcg_map = dict(zip(gcg_adv_data['id'], gcg_adv_data['perturbed_GCG_suffix']))
    
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
            elif sample == 'GCG':
                gcg_suffix = id_gcg_map[data["id"]]
                text = f"Question:\n{question}\nOption:\n{choices_text}" + gcg_suffix
                context = {"text": text, "image": image}
                aug_contexts[sample].append(context)
                continue
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

def construct_icl_prompt_for_image_heavy_with_answer(batch, dataset_nickname, model_nickname):
    """
    Constructs ICL-style prompts for image-heavy tasks, including answers.
    Each example consists of the related question, multiple-choice options, and the correct answer.
    """
    aug_contexts = defaultdict(list) 
    if "llava" in model_nickname:
        example_format = "Example:\nUSER: {text}ASSISTANT: {answer}\n"
    elif "blip" in model_nickname:
        example_format = "Example:\n{user_input}Answer: {answer}\n"
    
    for sample in ['related_text', 'origin', 'unrelated_text']:
        for data in batch:
            icl_context = ""
            question = data["question"]
            choices = data["multiple_choices"]
            gold_answer = data["multiple_choices_answer"]
            choices_text = ""
            
            # Format choices into text
            for c_name, c_content in choices.items():
                choices_text += f"{c_name}: {c_content}\n"

            # Load image
            try:
                image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
            except:
                image = Image.open(os.path.join(f"data/{dataset_nickname}/test", data["image"]))
            #origin setting
            origin_question = f"Question:\n{question}\nOption:\n{choices_text}\n"

            # related text
            interfere_facts = list(data.get("facts", {}).values())
            if interfere_facts:  # Ensure there are facts to choose from
                add_text = np.random.choice(interfere_facts)
                related_text_question = add_text + ' ' + question
                related_text_question = f"Question:\n{related_text_question}\nOption:\n{choices_text}\n"
            
            # unrelated text
            add_text = generate_unrelated_distraction_prompts()
            unrelated_text_question = add_text + question
            unrelated_text_question = f"Question:\n{unrelated_text_question}\nOption:\n{choices_text}\n"
            
            # Construct ICL example
            if sample == 'related_text':
                examples = [example_format.format(text=qes, answer=gold_answer) for qes in [origin_question, unrelated_text_question]]
                icl_context += ''.join(examples)
                text = related_text_question
    
            elif sample == 'origin': 
                examples = [example_format.format(text=qes, answer=gold_answer) for qes in [related_text_question, unrelated_text_question]]
                icl_context += ''.join(examples)
                text = origin_question

            elif sample == 'unrelated_text':
                examples = [example_format.format(text=qes, answer=gold_answer) for qes in [origin_question, related_text_question]]
                icl_context += ''.join(examples)
                text = unrelated_text_question

            context = {'text' : text, 'image' : image, 'icl_context' : icl_context}
            aug_contexts[sample].append(context)    

    return aug_contexts

def multi_img_icl_prompt_for_image_heavy(data, dataset_nickname, model_nickname):

    if "llava" in model_nickname:
        example_format = llava_example_format
    elif "blip" in model_nickname:
        example_format = blip_example_format

    context = {}
    for sample in ['related_text', 'origin', 'unrelated_text']:
        icl_context = ""
        question = data["question"]
        choices = data["multiple_choices"]
        gold_answer = data["multiple_choices_answer"]
        choices_text = ""
        for c_name, c_content in choices.items():
            choices_text += f"{c_name}: {c_content}\n"

        try:
            image = Image.open(os.path.join(f"data/{dataset_nickname}/images", data["image"]))
        except:
            image = Image.open(os.path.join(f"data/{dataset_nickname}/test", data["image"]))

        origin_question = f"Question:\n{question}\nOption:\n{choices_text}\n"

        # related text
        interfere_facts = list(data.get("facts", {}).values())
        if interfere_facts:  # Ensure there are facts to choose from
            add_text = np.random.choice(interfere_facts)
            related_text_question = add_text + ' ' + question
            related_text_question = f"Question:\n{related_text_question}\nOption:\n{choices_text}\n"
        
        # unrelated text
        add_text = generate_unrelated_distraction_prompts()
        unrelated_text_question = add_text + question
        unrelated_text_question = f"Question:\n{unrelated_text_question}\nOption:\n{choices_text}\n"
        
        # Construct ICL example
        if sample == 'related_text':
            examples = [example_format.format(text=qes, answer=gold_answer) for qes in [origin_question, unrelated_text_question]]
            icl_context += ''.join(examples)
            text = related_text_question

        elif sample == 'origin': 
            examples = [example_format.format(text=qes, answer=gold_answer) for qes in [related_text_question, unrelated_text_question]]
            icl_context += ''.join(examples)
            text = origin_question

        elif sample == 'unrelated_text':
            examples = [example_format.format(text=qes, answer=gold_answer) for qes in [origin_question, related_text_question]]
            icl_context += ''.join(examples)
            text = unrelated_text_question

        context[sample] = {'text' : text, 'image' : [image] * 3, 'icl_context' : icl_context}

    return context

# def denormalize_pixel_values(pixel_values, image_mean, image_std):
#     """
#     return CLIP encoded pixel values to original pixel values
#     """
#     pixel_values = pixel_values * torch.tensor(image_std, device=pixel_values.device).view(1, 3, 1, 1)
#     pixel_values = pixel_values + torch.tensor(image_mean, device=pixel_values.device).view(1, 3, 1, 1)
#     pixel_values = pixel_values * 255.0
#     pixel_values = torch.clamp(pixel_values, 0, 255).byte()
#     return pixel_values

# def FGSM_attack_image_generation_llava(contexts, model, processor, epsilon=0.01):
#     model.eval()
#     collactor = LLaVAOriginDataCollator(processor)
#     processed_inputs = collactor(contexts)
#     processed_inputs = SimpleNamespace(**processed_inputs)
#     # Only consider the gradient of pixel values
#     #processed_inputs = processed_inputs.to(model.device)
#     processed_inputs.input_ids = processed_inputs.input_ids.to(model.device)
#     processed_inputs.attention_mask = processed_inputs.attention_mask.to(model.device)
#     processed_inputs.pixel_values = processed_inputs.pixel_values.to(model.device)
#     processed_inputs.labels = processed_inputs.labels.to(model.device)

#     processed_inputs.pixel_values.requires_grad_()
#     for param in model.parameters():
#         param.requires_grad = False 

#     outputs = model(
#         input_ids=processed_inputs.input_ids,
#         attention_mask=processed_inputs.attention_mask,
#         pixel_values=processed_inputs.pixel_values,
#         labels=processed_inputs.labels,
#         return_dict=True,
#     )

#     loss = outputs.loss

#     model.zero_grad() # No need to record gradients for model parameters
#     loss.backward() 

#     # Collect the element-wise sign of the data gradient
#     pixel_grad = processed_inputs.pixel_values.grad.data.sign()
#     perturbed_pixels = processed_inputs.pixel_values + epsilon * pixel_grad
#     perturbed_pixels = torch.clamp(perturbed_pixels, -2, 2)
#     denormalized_pixels = denormalize_pixel_values(perturbed_pixels, image_mean, image_std)
#     return denormalized_pixels

# def PGD_attack_image_generation_llava(contexts, model, processor, alpha=0.1, epsilon=1, num_iter=10):
#     '''
#     Different from FGSM, alpha is the step size for each iteration, and epsilon is the maximum perturbation.
#     '''
#     model.eval()  
#     collactor = LLaVAOriginDataCollator(processor)

#     model.eval()
#     processed_inputs = collactor(contexts)
#     processed_inputs = SimpleNamespace(**processed_inputs)
#     # Only consider the gradient of pixel values
#     processed_inputs.input_ids = processed_inputs.input_ids.to(model.device)
#     processed_inputs.attention_mask = processed_inputs.attention_mask.to(model.device)
#     processed_inputs.pixel_values = processed_inputs.pixel_values.to(model.device)
#     processed_inputs.labels = processed_inputs.labels.to(model.device)
#     processed_inputs.pixel_values.requires_grad_()
#     for param in model.parameters():
#         param.requires_grad = False 

#     perturbed_pixels = processed_inputs.pixel_values.clone().detach()
#     # PGD attack process
#     for _ in range(num_iter):
#         perturbed_pixels.requires_grad = True  # reguire grad for perturbed pixels
#         outputs = model(
#             input_ids=processed_inputs.input_ids,
#             attention_mask=processed_inputs.attention_mask,
#             pixel_values=perturbed_pixels,
#             labels=processed_inputs.labels,
#             return_dict=True,
#         )

#         loss = outputs.loss
#         model.zero_grad()
#         loss.backward()

#         # Compute perturbation direction
#         pixel_grad = perturbed_pixels.grad.data.sign()
#         perturbed_pixels = perturbed_pixels + alpha * pixel_grad

#         # Limit Perturbation Range |x_adv - x| <= epsilon
#         perturbed_pixels = torch.clamp(
#             perturbed_pixels,
#             torch.max(processed_inputs.pixel_values - epsilon, torch.tensor(-2, device=perturbed_pixels.device)),
#             torch.min(processed_inputs.pixel_values + epsilon, torch.tensor(2, device=perturbed_pixels.device))
#         ).detach()

#     denormalized_pixels = denormalize_pixel_values(perturbed_pixels, image_mean, image_std)
#     return denormalized_pixels

# def FGSM_attack_image_generation_blip(images, texts, model, processor, epsilon=0.01):
#     model.eval()
#     prompt_prefixes = [text.split("Answer:")[0] for text in texts]
#     prefix_tokens = processor.tokenizer(
#         prompt_prefixes,
#         add_special_tokens=False,
#     )
#     prefix_lengths = [len(tokens) for tokens in prefix_tokens.input_ids]
    
#     qformer_inputs = processor(
#         text=texts,
#         images=images,
#         return_tensors="pt",
#         #padding="longest",
#         max_length=512, 
#         padding="max_length", 
#         truncation=True,
#         padding_side='right'
#     )

#     num_image_tokens = 32
    
#     labels = qformer_inputs.input_ids.clone()
#     for i, prefix_length in enumerate(prefix_lengths):
#         labels[i, :num_image_tokens + prefix_length + 2] = -100
#     labels[labels == processor.tokenizer.pad_token_id] = -100

#     qformer_inputs = qformer_inputs.to(model.device)
#     labels = labels.to(model.device)

#     qformer_inputs.pixel_values.requires_grad_()

#     for param in model.parameters():
#         param.requires_grad = False

#     outputs = model(
#         input_ids=qformer_inputs.input_ids,
#         attention_mask=qformer_inputs.attention_mask,
#         pixel_values=qformer_inputs.pixel_values,
#         qformer_input_ids=qformer_inputs.qformer_input_ids,
#         qformer_attention_mask=qformer_inputs.qformer_attention_mask,
#         labels=labels,
#         return_dict=True,
#     )

#     loss = outputs.loss

#     model.zero_grad()
#     loss.backward()

#     pixel_grad = qformer_inputs.pixel_values.grad.data.sign()
#     perturbed_pixels = qformer_inputs.pixel_values + epsilon * pixel_grad
#     perturbed_pixels = torch.clamp(perturbed_pixels, -2, 2)
#     denormalized_pixels = denormalize_pixel_values(perturbed_pixels, image_mean, image_std)
#     return denormalized_pixels

# def PGD_attack_image_generation_blip(images, texts, model, processor, alpha=0.1, epsilon=1, num_iter=10):
#     model.eval()
#     prompt_prefixes = [text.split("Answer:")[0] for text in texts]
#     prefix_tokens = processor.tokenizer(
#         prompt_prefixes,
#         add_special_tokens=False,
#     )
#     prefix_lengths = [len(tokens) for tokens in prefix_tokens.input_ids]
    
#     qformer_inputs = processor(
#         text=texts,
#         images=images,
#         return_tensors="pt",
#         #padding="longest",
#         max_length=512, 
#         padding="max_length", 
#         truncation=True,
#         padding_side='right'
#     )

#     num_image_tokens = 32
    
#     labels = qformer_inputs.input_ids.clone()
#     for i, prefix_length in enumerate(prefix_lengths):
#         labels[i, :num_image_tokens + prefix_length + 2] = -100
#     labels[labels == processor.tokenizer.pad_token_id] = -100

#     qformer_inputs = qformer_inputs.to(model.device)
#     labels = labels.to(model.device)

#     qformer_inputs.pixel_values.requires_grad_()

#     for param in model.parameters():
#         param.requires_grad = False

#     perturbed_pixels = qformer_inputs.pixel_values.clone().detach()
#     for _ in range(num_iter):
#         perturbed_pixels.requires_grad = True
#         outputs = model(
#             input_ids=qformer_inputs.input_ids,
#             attention_mask=qformer_inputs.attention_mask,
#             pixel_values=perturbed_pixels,
#             qformer_input_ids=qformer_inputs.qformer_input_ids,
#             qformer_attention_mask=qformer_inputs.qformer_attention_mask,
#             labels=labels,
#             return_dict=True,
#         )

#         loss = outputs.loss
#         model.zero_grad()
#         loss.backward()

#         pixel_grad = perturbed_pixels.grad.data.sign()
#         perturbed_pixels = perturbed_pixels + alpha * pixel_grad

#         perturbed_pixels = torch.clamp(
#             perturbed_pixels,
#             torch.max(qformer_inputs.pixel_values - epsilon, torch.tensor(-2, device=perturbed_pixels.device)),
#             torch.min(qformer_inputs.pixel_values + epsilon, torch.tensor(2, device=perturbed_pixels.device))
#         ).detach()

#     denormalized_pixels = denormalize_pixel_values(perturbed_pixels, image_mean, image_std)
#     return denormalized_pixels

# def GCG_attack_add_suffix_llava(context, model, processor, top_k=5, suffix_length=5, B=3, num_iter=10):
#     """
#     GCG attack: Find an adversarial suffix to append to the question, causing the model to misclassify the image.

#     Args:
#         model: Multi-modal language model (MLLM).
#         processor: Corresponding processor (LLaVA or InstructBLIP).
#         question: Original question (unaltered).
#         image: PIL image input.
#         choices: Dictionary of answer choices.
#         true_answer: The correct answer (ground truth).
#         suffix_length: Number of tokens in the adversarial suffix.
#         top_k: Number of top candidate tokens to explore at each step.
#         num_steps: Number of optimization steps.

#     Returns:
#         adversarial_question: The final adversarial question (original + optimized suffix).
#     """
#     model.eval()  
#     image, text = context[0][0], context[0][1]
#     # Preprocessing image, text for llava
#     collactor = LLaVAOriginDataCollator(processor)
#     # Edit input_ids and Preprocessing image, text for llava
#     processed_inputs = collactor(context)
#     processed_inputs = SimpleNamespace(**processed_inputs)
#     suffix_input_ids = torch.randint(0, processor.tokenizer.vocab_size, (1, suffix_length))
    
#     prefix_length = collactor.find_last_token_pos(processed_inputs.input_ids)
#     adv_input_ids = torch.cat((processed_inputs.input_ids[:, :prefix_length], suffix_input_ids,  processed_inputs.input_ids[:, prefix_length:]), dim=1) 
#     prefix_length_adv = collactor.find_last_token_pos(adv_input_ids)

#     # edit labels
#     adv_labels = adv_input_ids.clone().detach()
#     adv_labels[:, :prefix_length_adv] = -100
#     adv_labels[adv_labels == processor.tokenizer.pad_token_id] = -100

#     # edit attention mask
#     adv_atten_mask = torch.cat((torch.ones((1, int(suffix_length))), processed_inputs.attention_mask), dim=1)

#     adv_input_ids = adv_input_ids.to(model.device)
#     adv_atten_mask = adv_atten_mask.to(model.device)
#     adv_labels = adv_labels.to(model.device)
#     pixel_values = processed_inputs.pixel_values.to(model.device)

#     for param in model.parameters():
#         param.requires_grad = False 

#     embedding_layer = model.get_input_embeddings()

#     best_suffix = suffix_input_ids.clone().detach()
#     best_loss = float("-inf")

#     for step in range(num_iter):
#         adv_embeddings = embedding_layer(adv_input_ids)
#         adv_embeddings.requires_grad = True

#         outputs = model(input_ids=adv_input_ids, inputs_embeds=adv_embeddings, attention_mask=adv_atten_mask, pixel_values=pixel_values, labels=adv_labels, return_dict=True)

#         origin_loss = outputs.loss
#         model.zero_grad()
#         origin_loss.backward()

#         suffix_gradients = adv_embeddings.grad[:, prefix_length:prefix_length + suffix_length].abs().sum(dim=-1)
#         top_token_indices = torch.argsort(suffix_gradients, descending=True)[:, :top_k]
        
#         perturbed_suffix_candidates, losses_of_candidates = [], []
#         perturb_candidates = torch.topk(suffix_gradients[0], k=top_k, largest=True).indices

#         with torch.no_grad():
#             for _ in range(B):
#                 perturbed_suffix = suffix_input_ids.clone().detach()
#                 for idx in top_token_indices[0]:
#                     perturbed_suffix[0, idx] = random.choice(perturb_candidates)
#                 perturbed_suffix_candidates.append(perturbed_suffix)

#                 adv_input_ids = torch.cat((processed_inputs.input_ids[:, :prefix_length], perturbed_suffix,  processed_inputs.input_ids[:, prefix_length:]), dim=1).to(model.device)
#                 adv_embeddings = embedding_layer(adv_input_ids)
#                 outputs = model(input_ids=adv_input_ids, inputs_embeds=adv_embeddings, attention_mask=adv_atten_mask, pixel_values=pixel_values, labels=adv_labels, return_dict=True)

#                 new_loss = outputs.loss
#                 losses_of_candidates.append(new_loss.item())

#         losses_of_candidates.append(origin_loss.item()) 
#         perturbed_suffix_candidates.append(suffix_input_ids.clone().detach()) 
#         best_idx = torch.argmax(torch.tensor(losses_of_candidates))
#         best_candidate_loss = losses_of_candidates[best_idx]
#         if best_candidate_loss > best_loss:
#             best_loss = best_candidate_loss
#             best_suffix = perturbed_suffix_candidates[best_idx].clone().detach()

#         adv_input_ids.grad = None
#         model.zero_grad()
    
#     adversarial_suffix = processor.tokenizer.decode(best_suffix[0])

#     return adversarial_suffix

# def TextFooler_attack_image_generation_llava(images, texts, model, processor, alpha=0.1, epsilon=1, num_iter=10):
#     raise NotImplementedError

# def GCG_attack_image_generation_blip(images, texts, model, processor, epsilon=0.01):
#     raise NotImplementedError

# def TextFooler_attack_image_generation_blip(images, texts, model, processor, alpha=0.1, epsilon=1, num_iter=10):
#     raise NotImplementedError