import torch
import numpy as np

import torch.nn.functional as F
from torch import nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

from transformers import LlavaForConditionalGeneration
from transformers.utils.generic import ModelOutput
from .LLaVA_ad_attacks import *
from .LLaVA_consistency_regularization import *

@dataclass
class LlavaOutputwithConsistencyLoss(ModelOutput):
    """
    Extended class for Llava causal language model (or autoregressive) outputs.

    Args:
        consistency_loss (`torch.FloatTensor`, *optional*):
            The consistency loss calculated between augmented and original samples.
        original_loss (`torch.FloatTensor`, *optional*):
            The original loss calculated from the origin samples.
        (other arguments remain the same as original)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    consistency_loss: Optional[torch.FloatTensor] = None  # New field for consistency loss
    original_loss: Optional[torch.FloatTensor] = None     # New field for original loss

@dataclass
class LlavaOutputwithConsistencyRegandAdversarialReg(ModelOutput):
    """
    Extended class for Llava causal language model (or autoregressive) outputs.

    Args:
        consistency_loss (`torch.FloatTensor`, *optional*):
            The consistency loss calculated between augmented and original samples.
        original_loss (`torch.FloatTensor`, *optional*):
            The original loss calculated from the origin samples.
        (other arguments remain the same as original)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    consistency_loss: Optional[torch.FloatTensor] = None  # New field for consistency loss
    original_loss: Optional[torch.FloatTensor] = None     # New field for original loss
    perturbed_loss: Optional[torch.FloatTensor] = None     # New field for adversarial perturbed samples loss

class LLaVAForKLConsistencyPGDPerturbation(LLaVAForPGDPerturbation, LLaVAForKLdivergence):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        alpha=0.005,
        epsilon=0.03,
        pgd_steps=3,
        data_types=None,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, LlavaOutputwithConsistencyRegandAdversarialReg]:
        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            original_batch_size = origin["input_ids"].size(0)

            # Compute the loss with PGD perturbation(both origin sample and augmented samples)
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }

            combined_outputs = super().ADperturbed_forward(  
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,  
                input_ids=combined_inputs["input_ids"],
                pixel_values=combined_inputs["pixel_values"],
                attention_mask=combined_inputs["attention_mask"],
                labels=combined_inputs["labels"],
                return_dict=return_dict,
            )

            original_logits = combined_outputs.logits[:original_batch_size]  
            augmented_logits = combined_outputs.logits[original_batch_size:] # Including naive augmented samples and adversarial augmented samples
            matched_sample_idxs = torch.cat([augmented["sample_idxs"], origin["sample_idxs"], augmented["sample_idxs"]], dim=0)
            expanded_labels = torch.cat([augmented["labels"], origin["labels"], augmented["labels"]], dim=0)
            unique_sample_idxs = matched_sample_idxs.unique()

            for idx in unique_sample_idxs:
                # Find the origin samples and augmented samples that match the current idx
                sample_mask = matched_sample_idxs == idx
                current_augmented_logits = augmented_logits[sample_mask]
                current_augmented_labels = expanded_labels[sample_mask]
                
                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                augmented_answer_logits = current_augmented_logits.masked_select(mask_expanded).view(
                    current_augmented_logits.size(0),
                    -1,
                    current_augmented_logits.size(-1)
                )

                with torch.no_grad():
                    original_answer_mask = origin['labels'][idx] != -100
                    original_answer_logits = original_logits[idx, original_answer_mask, :]

                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1).clamp(min=1e-6)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

                kl_loss = F.kl_div(
                    F.log_softmax(original_answer_logits.unsqueeze(0).detach() / self.temp, dim=-1),
                    q_prob,
                    reduction='batchmean'
                )

                consistency_loss += kl_loss

            consistency_loss = consistency_loss / len(unique_sample_idxs)

            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=origin["input_ids"],
                pixel_values=origin["pixel_values"],
                attention_mask=origin["attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss + consistency_loss

        return LlavaOutputwithConsistencyRegandAdversarialReg(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            image_hidden_states=combined_outputs.image_hidden_states,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.origin_loss,
            perturbed_loss=combined_outputs.perturbed_loss,
        )
     
class LLaVAForJSdivergenceAndPGDPerturbation(LLaVAForJSdivergence, LLaVAForPGDPerturbation):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, LlavaOutputwithConsistencyLoss]:
        #Should call the forward method of LLaVAForJSdivergence
        return super().forward(origin=origin, augmented=augmented, return_dict=return_dict)

    def ADperturbed_forward(
        self,
        alpha=0.005,
        epsilon=0.03,
        pgd_steps=3,
        data_types=None,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, LlavaOutputwithConsistencyLoss]:
        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }
            # Compute the loss with PGD perturbation(both origin sample and augmented samples)
            combined_outputs = super().ADperturbed_forward(  
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,  
                input_ids=combined_inputs["input_ids"],
                pixel_values=combined_inputs["pixel_values"],
                attention_mask=combined_inputs["attention_mask"],
                labels=combined_inputs["labels"],
                return_dict=return_dict,
            )

            original_batch_size = origin["input_ids"].size(0)
            original_outputs_logits = combined_outputs.logits[:original_batch_size]
            augmented_outputs_logits = combined_outputs.logits[original_batch_size:]

            unique_sample_idxs = augmented["sample_idxs"].unique()
            for idx in unique_sample_idxs:
                sample_mask = augmented["sample_idxs"] == idx
                current_augmented_logits = augmented_outputs_logits[sample_mask]
                current_augmented_labels = augmented['labels'][sample_mask]

                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                augmented_answer_logits = current_augmented_logits.masked_select(mask_expanded).view(
                    current_augmented_logits.size(0),
                    -1,
                    current_augmented_logits.size(-1)
                )

                with torch.no_grad():
                    original_answer_mask = origin['labels'][idx] != -100
                    original_answer_logits = original_outputs_logits[idx, original_answer_mask, :]

                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1)
                q_prob = q_prob.clamp(min=self.epsilon)  
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True) 

                p_prob = F.softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1)
                p_prob = p_prob.clamp(min=self.epsilon)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                p_prob = p_prob.expand(q_prob.shape[0], -1, -1)

                m_prob = 0.5 * (p_prob + q_prob)
                kl_pm = F.kl_div(F.log_softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1),
                 m_prob, reduction='batchmean')
                kl_qm = F.kl_div(F.log_softmax(augmented_answer_logits / self.temp, dim=-1),
                                m_prob, reduction='batchmean')
                js_divergence = 0.5 * (kl_pm + kl_qm)

                consistency_loss += js_divergence

            consistency_loss = consistency_loss / len(unique_sample_idxs)
            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=origin["input_ids"],
                pixel_values=origin["pixel_values"],
                attention_mask=origin["attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)

        return LlavaOutputwithConsistencyLoss(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            image_hidden_states=combined_outputs.image_hidden_states,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.loss,
        )