import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Any

from transformers.utils.generic import ModelOutput

from .Qwen_ad_attacks import Qwen2_5_VLForPGDPerturbation
from .Qwen_consistency_regularization import *

@dataclass
class Qwen2_5_VLOutputWithConsistencyLossAndAdversarial(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    consistency_loss: Optional[torch.FloatTensor] = None
    original_loss: Optional[torch.FloatTensor] = None
    perturbed_loss: Optional[torch.FloatTensor] = None

class Qwen2_5_VLForKLConsistencyPGDPerturbation(Qwen2_5_VLForPGDPerturbation, Qwen2_5_VLForKLdivergence):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        self.consistency_loss_weight = 1.0
        self.temp = 1.0

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

    def forward(
        self, 
        alpha=0.005, 
        epsilon=0.03, 
        pgd_steps=3, 
        data_types=None,
        origin=None, 
        augmented=None, 
        return_dict=True
    ) -> Union[Tuple, Qwen2_5_VLOutputWithConsistencyLossAndAdversarial]:
        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            original_batch_size = origin["input_ids"].size(0)

            # Manually inject ADperturbed_forward output here
            combined_inputs = {
                k: torch.cat([origin[k], augmented[k]], dim=0)
                for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
            }

            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                image_grid_thw=combined_inputs["image_grid_thw"],
                labels=combined_inputs["labels"],
                return_dict=return_dict
            )

            original_logits = combined_outputs.logits[:original_batch_size]
            augmented_logits = combined_outputs.logits[original_batch_size:]
            
            matched_sample_idxs = torch.cat([augmented["sample_idxs"], origin["sample_idxs"], augmented["sample_idxs"]], dim=0)
            expanded_labels = torch.cat([augmented["labels"], origin["labels"], augmented["labels"]], dim=0)
            unique_sample_idxs = matched_sample_idxs.unique()

            for idx in unique_sample_idxs:
                sample_mask = matched_sample_idxs == idx
                current_augmented_logits = augmented_logits[sample_mask]
                current_augmented_labels = expanded_labels[sample_mask]

                augmented_answer_mask = current_augmented_labels != -100
                masked_logits = current_augmented_logits.masked_select(augmented_answer_mask.unsqueeze(-1))
                augmented_answer_logits = masked_logits.view(
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

            consistency_loss /= len(unique_sample_idxs)

            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only PGD loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                image_grid_thw=origin["image_grid_thw"],
                labels=origin["labels"],
                return_dict=return_dict
            )
            total_loss = combined_outputs.loss + consistency_loss

        return Qwen2_5_VLOutputWithConsistencyLossAndAdversarial(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=combined_outputs.rope_deltas,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.origin_loss,
            perturbed_loss=combined_outputs.perturbed_loss,
        )
    
class Qwen2_5_VLForJSdivergenceAndPGDPerturbation(Qwen2_5_VLForPGDPerturbation, Qwen2_5_VLForJSdivergence):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        self.consistency_loss_weight = 1.0
        self.temp = 1.0

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

    def forward(
        self, 
        alpha=0.005, 
        epsilon=0.03, 
        pgd_steps=3, 
        data_types=None,
        origin=None, 
        augmented=None, 
        return_dict=True
    ) -> Union[Tuple, Qwen2_5_VLOutputWithConsistencyLossAndAdversarial]:
        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            original_batch_size = origin["input_ids"].size(0)

            # Manually inject ADperturbed_forward output here
            combined_inputs = {
                k: torch.cat([origin[k], augmented[k]], dim=0)
                for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
            }

            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                image_grid_thw=combined_inputs["image_grid_thw"],
                labels=combined_inputs["labels"],
                return_dict=return_dict
            )

            original_logits = combined_outputs.logits[:original_batch_size]
            augmented_logits = combined_outputs.logits[original_batch_size:]
            
            matched_sample_idxs = torch.cat([augmented["sample_idxs"], origin["sample_idxs"], augmented["sample_idxs"]], dim=0)
            expanded_labels = torch.cat([augmented["labels"], origin["labels"], augmented["labels"]], dim=0)
            unique_sample_idxs = matched_sample_idxs.unique()

            for idx in unique_sample_idxs:
                sample_mask = matched_sample_idxs == idx
                current_augmented_logits = augmented_logits[sample_mask]
                current_augmented_labels = expanded_labels[sample_mask]

                augmented_answer_mask = current_augmented_labels != -100
                masked_logits = current_augmented_logits.masked_select(augmented_answer_mask.unsqueeze(-1))
                augmented_answer_logits = masked_logits.view(
                    current_augmented_logits.size(0), 
                    -1, 
                    current_augmented_logits.size(-1)
                )

                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1).clamp(min=self.epsilon)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

                with torch.no_grad():
                    original_answer_mask = origin['labels'][idx] != -100
                    original_answer_logits = original_logits[idx, original_answer_mask, :]

                p_prob = F.softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1).clamp(min=self.epsilon)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                p_prob = p_prob.expand(q_prob.size(0), -1, -1)

                m_prob = 0.5 * (p_prob + q_prob)

                kl_pm = F.kl_div(F.log_softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1), m_prob, reduction='batchmean')
                kl_qm = F.kl_div(F.log_softmax(augmented_answer_logits / self.temp, dim=-1), m_prob, reduction='batchmean')
                js_div = 0.5 * (kl_pm + kl_qm)

                consistency_loss += js_div

            consistency_loss /= len(unique_sample_idxs)

            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: JS loss is NaN/Inf, skipping...")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                image_grid_thw=origin["image_grid_thw"],
                labels=origin["labels"],
                return_dict=return_dict
            )
            total_loss = combined_outputs.loss + consistency_loss

        if hasattr(self, "training") and self.training:
            print(f"[Qwen-JS] Total: {total_loss.item():.4f} | Origin: {combined_outputs.loss.item():.4f} | Consistency(JS): {consistency_loss.item():.4f}")
            
        return Qwen2_5_VLOutputWithConsistencyLossAndAdversarial(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=combined_outputs.rope_deltas,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.origin_loss,
            perturbed_loss=combined_outputs.perturbed_loss,
        )