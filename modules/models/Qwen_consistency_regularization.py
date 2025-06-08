import torch

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.utils.generic import ModelOutput

@dataclass
class Qwen2_5_VLOutputWithConsistencyLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    consistency_loss: Optional[torch.FloatTensor] = None
    original_loss: Optional[torch.FloatTensor] = None

    
class Qwen2_5_VLForKLdivergence(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        self.consistency_loss_weight = 1.0
        self.temp = 1.0  # temperature for softmax

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True,
    ) -> Union[Tuple, Qwen2_5_VLOutputWithConsistencyLoss]:
        consistency_loss = torch.tensor(0.0, device=self.device)
        
        if augmented is not None:
            combined_inputs = {
                k: torch.cat([origin[k], augmented[k]], dim=0)
                for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
                if k in origin and k in augmented
            }

            combined_outputs = super().forward(
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                image_grid_thw=combined_inputs["image_grid_thw"],
                labels=combined_inputs["labels"],
                return_dict=return_dict
            )

            original_batch_size = origin["input_ids"].size(0)
            original_logits = combined_outputs.logits[:original_batch_size]
            augmented_logits = combined_outputs.logits[original_batch_size:]

            for idx in augmented["sample_idxs"].unique():
                aug_mask = augmented["sample_idxs"] == idx
                current_aug_logits = augmented_logits[aug_mask]
                current_aug_labels = augmented["labels"][aug_mask]

                aug_answer_mask = current_aug_labels != -100
                masked_logits = current_aug_logits.masked_select(aug_answer_mask.unsqueeze(-1))
                aug_answer_logits = masked_logits.view(current_aug_logits.size(0), -1, current_aug_logits.size(-1))

                with torch.no_grad():
                    origin_answer_mask = origin["labels"][idx] != -100
                    origin_answer_logits = original_logits[idx, origin_answer_mask, :]

                q_prob = F.softmax(aug_answer_logits / self.temp, dim=-1).clamp(min=self.epsilon)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

                p_log_prob = F.log_softmax(origin_answer_logits.unsqueeze(0).detach() / self.temp, dim=-1)
                kl_loss = F.kl_div(p_log_prob, q_prob, reduction="batchmean")
                consistency_loss += kl_loss

            consistency_loss /= len(augmented["sample_idxs"].unique())
            total_loss = combined_outputs.loss + self.consistency_loss_weight * consistency_loss

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                image_grid_thw=origin["image_grid_thw"],
                labels=origin["labels"],
                return_dict=True
            )
            total_loss = combined_outputs.loss

        if hasattr(self, "training") and self.training:
            print(f"[Qwen-KL] Total: {total_loss.item():.4f} | Origin: {combined_outputs.loss.item():.4f} | Consistency: {consistency_loss.item():.4f}")

        return Qwen2_5_VLOutputWithConsistencyLoss(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=combined_outputs.rope_deltas,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.loss,
        )
    
class Qwen2_5_VLForJSdivergence(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        self.consistency_loss_weight = 1.0
        self.temp = 1.0  # temperature for softmax

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True,
    ) -> Union[Tuple, Qwen2_5_VLOutputWithConsistencyLoss]:

        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            combined_inputs = {
                k: torch.cat([origin[k], augmented[k]], dim=0)
                for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
                if k in origin and k in augmented
            }

            combined_outputs = super().forward(
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                image_grid_thw=combined_inputs["image_grid_thw"],
                labels=combined_inputs["labels"],
                return_dict=return_dict
            )

            original_batch_size = origin["input_ids"].size(0)
            original_logits = combined_outputs.logits[:original_batch_size]
            augmented_logits = combined_outputs.logits[original_batch_size:]

            unique_sample_idxs = augmented["sample_idxs"].unique()

            for idx in unique_sample_idxs:
                mask = augmented["sample_idxs"] == idx
                current_aug_logits = augmented_logits[mask]
                current_aug_labels = augmented["labels"][mask]

                aug_answer_mask = current_aug_labels != -100
                masked_logits = current_aug_logits.masked_select(aug_answer_mask.unsqueeze(-1))
                aug_answer_logits = masked_logits.view(current_aug_logits.size(0), -1, current_aug_logits.size(-1))

                q_prob = F.softmax(aug_answer_logits / self.temp, dim=-1).clamp(min=self.epsilon)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

                with torch.no_grad():
                    origin_answer_mask = origin["labels"][idx] != -100
                    origin_answer_logits = original_logits[idx, origin_answer_mask, :]

                p_prob = F.softmax(origin_answer_logits.unsqueeze(0) / self.temp, dim=-1).clamp(min=self.epsilon)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                p_prob = p_prob.expand(q_prob.size(0), -1, -1)

                m_prob = 0.5 * (p_prob + q_prob)

                kl_pm = F.kl_div(F.log_softmax(origin_answer_logits.unsqueeze(0) / self.temp, dim=-1), m_prob, reduction='batchmean')
                kl_qm = F.kl_div(F.log_softmax(aug_answer_logits / self.temp, dim=-1), m_prob, reduction='batchmean')
                js_div = 0.5 * (kl_pm + kl_qm)

                consistency_loss += js_div

            consistency_loss = consistency_loss / len(unique_sample_idxs)

            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + self.consistency_loss_weight * consistency_loss
            else:
                print("Warning: JS loss is NaN/Inf, skipping...")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                image_grid_thw=origin["image_grid_thw"],
                labels=origin["labels"],
                return_dict=True
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self, "training") and self.training:
            print(f"[Qwen-JS] Total: {total_loss.item():.4f} | Origin: {combined_outputs.loss.item():.4f} | Consistency(JS): {consistency_loss.item():.4f}")

        return Qwen2_5_VLOutputWithConsistencyLoss(
            loss=total_loss,
            logits=combined_outputs.logits,
            past_key_values=combined_outputs.past_key_values,
            hidden_states=combined_outputs.hidden_states,
            attentions=combined_outputs.attentions,
            rope_deltas=combined_outputs.rope_deltas,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.loss,
        )