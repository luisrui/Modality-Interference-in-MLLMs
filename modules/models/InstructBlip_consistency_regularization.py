import torch

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any

from transformers import InstructBlipForConditionalGeneration
from transformers.utils.generic import ModelOutput

@dataclass
class InstructBlipOutputWithConsistencyLoss(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    consistency_loss: Optional[torch.FloatTensor] = None
    original_loss: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    
class InstructBlipForKLdivergence(InstructBlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        
    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, InstructBlipOutputWithConsistencyLoss]:

        consistency_loss = torch.tensor(0.0, device=self.device)
        if augmented is not None:
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "qformer_input_ids": torch.cat([origin["qformer_input_ids"], augmented["qformer_input_ids"]], dim=0),
                "qformer_attention_mask": torch.cat([origin["qformer_attention_mask"], augmented["qformer_attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }
            combined_outputs = super().forward(
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                qformer_input_ids=combined_inputs["qformer_input_ids"],
                qformer_attention_mask=combined_inputs["qformer_attention_mask"],
                labels=combined_inputs["labels"],
                return_dict=return_dict,
            )

            original_batch_size = origin["input_ids"].size(0)
            original_outputs_logits = combined_outputs.logits[:original_batch_size]
            augmented_outputs_logits = combined_outputs.logits[original_batch_size:]

            # Calculate KL divergence for each unique sample
            unique_sample_idxs = augmented["sample_idxs"].unique()
            
            for idx in unique_sample_idxs:
                mask = augmented["sample_idxs"] == idx
                current_augmented_logits = augmented_outputs_logits[mask]
                current_augmented_labels = augmented['labels'][mask]

                # Get valid answer tokens (non-padding)
                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                masked_logits = current_augmented_logits.masked_select(mask_expanded)
                augmented_answer_logits = masked_logits.view(
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

                # Compute KL divergence
                kl_loss = F.kl_div(
                    F.log_softmax(original_answer_logits.unsqueeze(0).detach() / self.temp, dim=-1),
                    q_prob,
                    reduction='batchmean'
                )

                # torch.cuda.empty_cache()
                consistency_loss += kl_loss

            consistency_loss /= len(unique_sample_idxs)
            total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                qformer_input_ids=origin["qformer_input_ids"],
                qformer_attention_mask=origin["qformer_attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self, 'training') and self.training:
            print(f"Total loss: {total_loss.item():.4f}, "
                f"Original loss: {combined_outputs.loss.item():.4f}, "
                f"Consistency loss: {consistency_loss.item():.4f}")

        return InstructBlipOutputWithConsistencyLoss(
            loss=total_loss,
            logits=combined_outputs.logits,
            vision_outputs=combined_outputs.vision_outputs,
            qformer_outputs=combined_outputs.qformer_outputs,
            language_model_outputs=combined_outputs.language_model_outputs,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.loss,
        )

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

class InstructBlipForJSdivergence(InstructBlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        
    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, InstructBlipOutputWithConsistencyLoss]:

        consistency_loss = torch.tensor(0.0, device=self.device)
        if augmented is not None:
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "qformer_input_ids": torch.cat([origin["qformer_input_ids"], augmented["qformer_input_ids"]], dim=0),
                "qformer_attention_mask": torch.cat([origin["qformer_attention_mask"], augmented["qformer_attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }
            combined_outputs = super().forward(
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                pixel_values=combined_inputs["pixel_values"],
                qformer_input_ids=combined_inputs["qformer_input_ids"],
                qformer_attention_mask=combined_inputs["qformer_attention_mask"],
                labels=combined_inputs["labels"],
                return_dict=return_dict,
            )

            original_batch_size = origin["input_ids"].size(0)
            original_outputs_logits = combined_outputs.logits[:original_batch_size]
            augmented_outputs_logits = combined_outputs.logits[original_batch_size:]

            # Calculate KL divergence for each unique sample
            unique_sample_idxs = augmented["sample_idxs"].unique()
            
            for idx in unique_sample_idxs:
                mask = augmented["sample_idxs"] == idx
                current_augmented_logits = augmented_outputs_logits[mask]
                current_augmented_labels = augmented['labels'][mask]

                # Get valid answer tokens (non-padding)
                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                masked_logits = current_augmented_logits.masked_select(mask_expanded)
                augmented_answer_logits = masked_logits.view(
                    current_augmented_logits.size(0),
                    -1,
                    current_augmented_logits.size(-1)
                )

                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1)
                q_prob = q_prob.clamp(min=self.epsilon)
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True)

                with torch.no_grad():
                    original_answer_mask = origin['labels'][idx] != -100
                    original_answer_logits = original_outputs_logits[idx, original_answer_mask, :]

                p_prob = F.softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1)
                p_prob = p_prob.clamp(min=self.epsilon)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                p_prob = p_prob.expand(q_prob.shape[0], -1, -1)  # expand to match q_prob shape

                m_prob = 0.5 * (p_prob + q_prob)

                kl_pm = F.kl_div(F.log_softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1), m_prob, reduction='batchmean')
                kl_qm = F.kl_div(F.log_softmax(augmented_answer_logits / self.temp, dim=-1), m_prob, reduction='batchmean')
                js_loss = 0.5 * (kl_pm + kl_qm)

                consistency_loss += js_loss

            consistency_loss /= len(unique_sample_idxs)
            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                pixel_values=origin["pixel_values"],
                qformer_input_ids=origin["qformer_input_ids"],
                qformer_attention_mask=origin["qformer_attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self, 'training') and self.training:
            print(f"Total loss: {total_loss.item():.4f}, "
                f"Original loss: {combined_outputs.loss.item():.4f}, "
                f"Consistency loss: {consistency_loss.item():.4f}")

        return InstructBlipOutputWithConsistencyLoss(
            loss=total_loss,
            logits=combined_outputs.logits,
            vision_outputs=combined_outputs.vision_outputs,
            qformer_outputs=combined_outputs.qformer_outputs,
            language_model_outputs=combined_outputs.language_model_outputs,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.loss,
        )

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature