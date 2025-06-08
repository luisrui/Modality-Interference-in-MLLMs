import torch
import numpy as np

import torch.nn.functional as F
from torch import nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

from transformers import LlavaForConditionalGeneration
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
class LlavaCausalLMOutputWithAdvPerturbation(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    origin_loss : Optional[torch.FloatTensor] = None
    perturbed_loss : Optional[torch.FloatTensor] = None

class LLaVAForRGPerturbation(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def ADperturbed_forward(
        self,
        sigma: float,
        data_types: List[str] = None,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithAdvPerturbation]:
        """
        Forward pass for the RGPerturbation model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        special_visual_token_mask = (input_ids == self.config.image_token_index).to(inputs_embeds.device)
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_visual_token_mask.unsqueeze(-1), 
                image_features.view(-1, image_features.shape[-1])
            )
    
        perturbation_noise = torch.randn_like(inputs_embeds) * sigma
        perturbation_mask = generate_modality_mask(data_types, special_visual_token_mask, labels)  
        perturbation_mask = perturbation_mask.unsqueeze(-1)
        perturbed_inputs_embeds = inputs_embeds + (perturbation_noise * perturbation_mask.float())
        perturbed_inputs_embeds = perturbed_inputs_embeds.to(inputs_embeds.dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=perturbed_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithAdvPerturbation(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    
class LLaVAForPGDPerturbation(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def ADperturbed_forward(
        self,
        alpha,
        epsilon,
        pgd_steps,
        data_types: List[str] = None,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithAdvPerturbation]:
        """
        Forward pass for the RGPerturbation model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        special_visual_token_mask = (input_ids == self.config.image_token_index).to(inputs_embeds.device)
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_visual_token_mask.unsqueeze(-1), 
                image_features.view(-1, image_features.shape[-1])
            )
        perturbation_mask = generate_modality_mask(data_types, special_visual_token_mask, labels).unsqueeze(-1)  

        # Apply PGD perturbation
        for param in self.language_model.parameters():
            param.requires_grad = False

        perturbation_noise = (torch.rand_like(inputs_embeds) * 2 - 1) * epsilon
        input_embeds_pgd = inputs_embeds.clone().detach()
        for _ in range(pgd_steps):
            perturbation_noise = perturbation_noise.detach().clone().requires_grad_(True)  # Detach, clone, and enable grad
            perturbation_noise.retain_grad()
            perturbed_inputs_embeds = input_embeds_pgd + perturbation_noise * perturbation_mask.float()
            perturbed_inputs_embeds = perturbed_inputs_embeds.to(input_embeds_pgd.dtype)

            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=perturbed_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )
            logits = outputs[0]

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

            loss.backward()

            perturbation_noise = perturbation_noise + alpha * perturbation_noise.grad
            perturbation_noise = torch.clamp(perturbation_noise, -epsilon, epsilon)

        perturbation_noise = perturbation_noise.detach().clone().requires_grad_(False)
        perturbed_inputs_embeds = input_embeds_pgd + (perturbation_noise * perturbation_mask.float())
        perturbed_inputs_embeds = perturbed_inputs_embeds.to(inputs_embeds.dtype)

        combined_inputs_embeds = torch.cat([inputs_embeds, perturbed_inputs_embeds], dim=0)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
        # combined_labels = torch.cat([labels, labels], dim=0) if labels is not None else None

        for param in self.language_model.parameters():
            param.requires_grad = True

        combined_outputs = self.language_model(
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=combined_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        combined_logits = combined_outputs[0]

        if labels is not None:
            # Flatten the tokens
            batch_size = input_ids.shape[0]
            original_logits = combined_logits[:batch_size]
            perturbed_logits = combined_logits[batch_size:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -(original_logits.shape[1] - 1) :].to(original_logits.device)
                shift_labels = labels[..., 1:].contiguous()
                shift_original_logits = original_logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_perturbed_logits = perturbed_logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            original_loss = loss_fct(shift_original_logits.view(-1, shift_original_logits.size(-1)), shift_labels.view(-1))
            perturbed_loss = loss_fct(shift_perturbed_logits.view(-1, shift_perturbed_logits.size(-1)), shift_labels.view(-1))
            loss = original_loss + perturbed_loss
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithAdvPerturbation(
            loss=loss,
            logits=combined_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            origin_loss=original_loss if loss is not None else None,
            perturbed_loss=perturbed_loss if loss is not None else None
        )

