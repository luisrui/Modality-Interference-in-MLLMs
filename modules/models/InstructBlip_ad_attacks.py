import torch
import numpy as np

import torch.nn.functional as F
from torch import nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any

from transformers import InstructBlipForConditionalGeneration
from transformers.utils.generic import ModelOutput

def generate_modality_mask(data_types: List[str], visual_mask: torch.BoolTensor, labels: torch.LongTensor):
    """
    Generate a mask for modality perturbation
    """
    perturbation_mask = torch.zeros_like(visual_mask, dtype=torch.bool)

    for i, task_type in enumerate(data_types):
        if task_type == "image_heavy":
            perturbation_mask[i] = ~visual_mask[i] # only perturb text tokens
            perturbation_mask[i][labels[i] != -100] = False # no perturbation for label tokens
        elif task_type == "text_heavy":
            perturbation_mask[i] = visual_mask[i]  # only perturb image tokens
        elif task_type == "VQA":
            perturbation_mask[i] = False  # no perturbation for VQA

    return perturbation_mask

@dataclass
class InstructBlipOutputWithAdvPerturbation(ModelOutput):

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    origin_loss: Optional[Tuple[torch.FloatTensor]] = None
    perturbed_loss : Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

# class QueryTokensModule(nn.Module):
#     def __init__(self, num_query_tokens: int, hidden_size: int):
#         super().__init__()
#         self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

#     def forward(self, batch_size: int) -> torch.Tensor:
#         return self.query_tokens.expand(batch_size, -1, -1)
        
class InstructBlipForPGDPerturbation(InstructBlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.query_tokens_for_pgd = self.query_tokens.data.clone()
        self.query_tokens_for_pgd.requires_grad = False

    def ADperturbed_forward(
        self,
        alpha: float,
        epsilon: float,
        pgd_steps: int,
        data_types: List[str],
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):  
        # print("[DEBUG] query_tokens shape:", self.query_tokens_for_pgd.shape)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. vision encoder
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # 2. QFormer（freeze 后仍 forward）
        query_tokens = self.query_tokens_for_pgd.expand(image_embeds.shape[0], -1, -1).to(image_embeds.device)
        #query_tokens = self.query_tokens_module.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        qformer_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = qformer_outputs[0][:, : query_tokens.size(1), :]
        vision_tokens = self.language_projection(query_output)
        
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if getattr(self.config, "image_token_index", None) is not None:
            special_visual_mask = (input_ids == self.config.image_token_index).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[special_visual_mask.unsqueeze(-1).expand_as(inputs_embeds)] = vision_tokens.flatten()
        else:
            raise ValueError("Missing image_token_index in config, cannot apply vision embedding replacement.")

        perturbation_mask = generate_modality_mask(data_types, special_visual_mask, labels).unsqueeze(-1)

        for param in self.language_model.parameters():
            param.requires_grad = False
        
        perturbation_noise = (torch.rand_like(inputs_embeds) * 2 - 1) * epsilon
        input_embeds_pgd = inputs_embeds.detach().clone()
        for _ in range(pgd_steps):
            perturbation_noise = perturbation_noise.detach().clone().requires_grad_(True)
            perturbation_noise.retain_grad()
            perturbed_inputs_embeds = input_embeds_pgd + perturbation_noise * perturbation_mask.float()
            perturbed_inputs_embeds = perturbed_inputs_embeds.to(input_embeds_pgd.dtype)

            outputs = self.language_model(
                inputs_embeds=perturbed_inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            logits = outputs[0]

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

            loss.backward()

            perturbation_noise = perturbation_noise + alpha * perturbation_noise.grad
            perturbation_noise = torch.clamp(perturbation_noise, -epsilon, epsilon)
        
        for param in self.language_model.parameters():
            param.requires_grad = True

        perturbation_noise = perturbation_noise.detach().clone().requires_grad_(False)
        perturbed_inputs_embeds = input_embeds_pgd + (perturbation_noise * perturbation_mask.float())
        perturbed_inputs_embeds = perturbed_inputs_embeds.to(inputs_embeds.dtype)
        
        combined_inputs_embeds = torch.cat([inputs_embeds, perturbed_inputs_embeds], dim=0)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        #combined_labels = torch.cat([labels, labels], dim=0)
        
        combined_outputs = self.language_model(
            inputs_embeds=combined_inputs_embeds,
            attention_mask=combined_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        combined_logits = combined_outputs[0]

        batch_size = input_ids.shape[0]
        logits_ori = combined_logits[:batch_size]
        logits_adv = combined_logits[batch_size:]

        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, -(logits_ori.shape[1] - 1) :].to(logits_ori.device)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits_ori = logits_ori[..., :-1, :][shift_attention_mask != 0].contiguous()
            shift_logits_adv = logits_adv[..., :-1, :][shift_attention_mask != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss_ori = loss_fct(shift_logits_ori.view(-1, shift_logits_ori.size(-1)), shift_labels.view(-1))
        loss_adv = loss_fct(shift_logits_adv.view(-1, shift_logits_adv.size(-1)), shift_labels.view(-1))
        loss = loss_ori + loss_adv

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return InstructBlipOutputWithAdvPerturbation(
            loss=loss,
            logits=combined_logits,
            vision_outputs=vision_outputs,
            qformer_outputs=qformer_outputs,
            language_model_outputs=combined_outputs,
            origin_loss=loss_ori,
            perturbed_loss=loss_adv,
        )