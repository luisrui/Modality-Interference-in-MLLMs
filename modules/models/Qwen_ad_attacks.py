import torch
import numpy as np
import torch.nn.functional as F

from torch import nn as nn
from torch.amp import autocast
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.utils.generic import ModelOutput
from torch.utils.checkpoint import checkpoint
from functools import partial

# checkpoint.use_reentrant = False

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

def enable_non_reentrant(model):
    if hasattr(model, "_gradient_checkpointing_func"):
        model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

def enable_reentrant(model):
    if hasattr(model, "_gradient_checkpointing_func"):
        model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)

@dataclass
class Qwen2_5_VLOutputWitAdvPerturbation(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    origin_loss: Optional[torch.FloatTensor] = None
    perturbed_loss: Optional[torch.FloatTensor] = None

class Qwen2_5_VLForPGDPerturbation(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self._check_non_float_trainable_params()

    def _check_non_float_trainable_params(self):
        print("\n[Debug] Checking non-float parameters that require gradients:")
        for name, param in self.named_parameters():
            if param.requires_grad and not param.dtype.is_floating_point:
                print(f"  [Warning] Param `{name}` has dtype {param.dtype} but requires_grad=True")

    def ADperturbed_forward(
        self,
        alpha,
        epsilon,
        pgd_steps,
        data_types: List[str],
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Qwen2_5_VLOutputWitAdvPerturbation:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

        # PGD perturbation
        perturbation_mask = generate_modality_mask(data_types, (input_ids == self.config.image_token_id), labels).unsqueeze(-1)
        pgd_noise = (torch.rand_like(inputs_embeds) * 2 - 1) * epsilon
        input_embeds_pgd = inputs_embeds.clone().detach()

        # for param in self.model.parameters():
        #     param.requires_grad = False
        enable_non_reentrant(self.model)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(pgd_steps):
            pgd_noise = pgd_noise.detach().requires_grad_(True)
            perturbed = input_embeds_pgd + pgd_noise * perturbation_mask.float()
            with autocast(device_type="cuda"):
                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    inputs_embeds=perturbed,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=cache_position
                )
                logits = self.lm_head(outputs[0])
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels)
            #loss.backward()
            grad = torch.autograd.grad(loss, pgd_noise, retain_graph=False, create_graph=False)[0]
            pgd_noise = torch.clamp(pgd_noise + alpha * grad, -epsilon, epsilon).detach()
            del loss, grad, logits, outputs
            torch.cuda.empty_cache()
        enable_reentrant(self.model)
        # for param in self.model.parameters():
        #     param.requires_grad = True

        pgd_noise = pgd_noise.detach().clone().requires_grad_(False)
        perturbed_inputs_embeds = input_embeds_pgd + (pgd_noise * perturbation_mask.float())
        perturbed_inputs_embeds = perturbed_inputs_embeds.to(inputs_embeds.dtype)
        
        # Combine original and perturbed
        combined_inputs_embeds = torch.cat([inputs_embeds, perturbed_inputs_embeds], dim=0)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        combined_position_ids = torch.cat([position_ids, position_ids], dim=1)

        #print(combined_position_ids.shape, combined_attention_mask.shape, combined_inputs_embeds.shape)
        combined_outputs = self.model(
            input_ids=None,
            position_ids=combined_position_ids,
            attention_mask=combined_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=combined_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        logits = self.lm_head(combined_outputs[0])

        # Split and compute losses
        B = labels.shape[0]
        logits_ori, logits_adv = logits[:B], logits[B:]

        shift_logits_ori = logits_ori[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_logits_adv = logits_adv[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)

        loss_fct = nn.CrossEntropyLoss()
        loss_ori = loss_fct(shift_logits_ori, shift_labels)
        loss_adv = loss_fct(shift_logits_adv, shift_labels)
        total_loss = loss_ori + loss_adv

        return Qwen2_5_VLOutputWitAdvPerturbation(
            loss=total_loss,
            logits=logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=self.rope_deltas,
            origin_loss=loss_ori,
            perturbed_loss=loss_adv,
        )
        
class Qwen2_5_VLForRGPerturbation(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def ADperturbed_forward(
        self,
        sigma: float,
        data_types: List[str],
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = (input_ids == self.config.image_token_id)
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds.to(inputs_embeds.dtype))

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

        # === Add Gaussian Noise ===
        perturbation_mask = generate_modality_mask(data_types, (input_ids == self.config.image_token_id), labels).unsqueeze(-1)
        noise = torch.randn_like(inputs_embeds) * sigma
        perturbed_inputs_embeds = inputs_embeds + noise * perturbation_mask.float()
        perturbed_inputs_embeds = perturbed_inputs_embeds.to(inputs_embeds.dtype)

        # Combine original and perturbed
        combined_inputs_embeds = torch.cat([inputs_embeds, perturbed_inputs_embeds], dim=0)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        combined_position_ids = torch.cat([position_ids, position_ids], dim=1)

        combined_outputs = self.model(
            input_ids=None,
            position_ids=combined_position_ids,
            attention_mask=combined_attention_mask,
            past_key_values=None,
            inputs_embeds=combined_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        logits = self.lm_head(combined_outputs[0])

        # Compute loss
        B = labels.shape[0]
        logits_ori, logits_noise = logits[:B], logits[B:]

        shift_logits_ori = logits_ori[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_logits_noise = logits_noise[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)

        loss_fct = nn.CrossEntropyLoss()
        loss_ori = loss_fct(shift_logits_ori, shift_labels)
        loss_noise = loss_fct(shift_logits_noise, shift_labels)
        total_loss = loss_ori + loss_noise

        return Qwen2_5_VLOutputWitAdvPerturbation(
            loss=total_loss,
            logits=logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=self.rope_deltas,
            origin_loss=loss_ori,
            perturbed_loss=loss_noise,
        )