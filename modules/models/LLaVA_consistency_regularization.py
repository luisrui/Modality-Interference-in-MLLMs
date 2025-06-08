import torch

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

from transformers import LlavaForConditionalGeneration
from transformers.utils.generic import ModelOutput
    
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

class LLaVAForKLdivergence(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):  
                combined_outputs = super().forward(
                    input_ids=combined_inputs["input_ids"],
                    pixel_values=combined_inputs["pixel_values"],
                    attention_mask=combined_inputs["attention_mask"],
                    labels=combined_inputs["labels"],
                    return_dict=return_dict,
                    output_hidden_states=True 
                )

            original_batch_size = origin["input_ids"].size(0)
            original_outputs_logits = combined_outputs.logits[:original_batch_size]
            augmented_outputs_logits = combined_outputs.logits[original_batch_size:]

            unique_sample_idxs = augmented["sample_idxs"].unique()
            
            for idx in unique_sample_idxs:
                sample_mask = augmented["sample_idxs"] == idx
                current_augmented_logits = augmented_outputs_logits[sample_mask]
                current_augmented_labels = augmented['labels'][sample_mask]

                #augmented_cls_loss += augmented_outputs.loss
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

                epsilon = 1e-8 if augmented_answer_logits.dtype == torch.float32 else 1e-6
                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1)
                q_prob = q_prob.clamp(min=epsilon)  
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True) 

                kl_loss = F.kl_div(
                    F.log_softmax(original_answer_logits.unsqueeze(0).detach() / self.temp, dim=-1),
                    q_prob,
                    reduction='batchmean'
                )

                #torch.cuda.empty_cache()
                consistency_loss += kl_loss

            consistency_loss = consistency_loss / len(unique_sample_idxs)
            #augmented_cls_loss /= len(unique_sample_idxs)
            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                pixel_values=origin["pixel_values"],
                attention_mask=origin["attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        # if hasattr(self, 'training') and self.training:
        #     print(f"Total loss: {total_loss.item():.4f}, "
        #           f"Original loss: {combined_outputs.loss.item():.4f}, "
        #           f"Consistency loss: {consistency_loss.item():.4f}")
            
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

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

class LLaVAForJSdivergence(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        
    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, LlavaOutputwithConsistencyLoss]:
        
        consistency_loss = torch.tensor(0.0, device=self.device)

        # augmented = None #debug
        # process augmented samples
        if augmented is not None:
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16): 
                combined_outputs = super().forward(
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

                #augmented_cls_loss += augmented_outputs.loss
                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                augmented_answer_logits = current_augmented_logits.masked_select(mask_expanded).view(
                    current_augmented_logits.size(0),
                    -1,
                    current_augmented_logits.size(-1)
                )
                # compute JS divergence of origin and augment samples
                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1)
                q_prob = q_prob.clamp(min=self.epsilon)  
                q_prob = q_prob / q_prob.sum(dim=-1, keepdim=True) 

                original_answer_mask = origin['labels'][idx] != -100
                original_answer_logits = original_outputs_logits[idx, original_answer_mask, :]
                p_prob = F.softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1)
                p_prob = p_prob.clamp(min=self.epsilon)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True)
                p_prob = p_prob.expand(q_prob.shape[0], -1, -1)  # expand to match q_prob shape
                
                m_prob = 0.5 * (p_prob + q_prob)
                kl_pm = F.kl_div(F.log_softmax(original_answer_logits.unsqueeze(0) / self.temp, dim=-1), m_prob, reduction='batchmean')
                kl_qm = F.kl_div(F.log_softmax(augmented_answer_logits / self.temp, dim=-1), m_prob, reduction='batchmean')
                js_divergence = 0.5 * (kl_pm + kl_qm)

                consistency_loss += js_divergence

            consistency_loss = consistency_loss / len(unique_sample_idxs)
            #augmented_cls_loss /= len(unique_sample_idxs)
            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().forward(
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
    
    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature

class LLaVAForMSEdivergence(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6

    def forward(
        self,
        origin=None,
        augmented=None,
        return_dict=True
    ) -> Union[Tuple, LlavaOutputwithConsistencyLoss]:

        consistency_loss = torch.tensor(0.0, device=self.device)

        # augmented = None #debug
        # process augmented samples
        if augmented is not None:
            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                combined_outputs = super().forward(
                    input_ids=combined_inputs["input_ids"],
                    pixel_values=combined_inputs["pixel_values"],
                    attention_mask=combined_inputs["attention_mask"],
                    labels=combined_inputs["labels"],
                    return_dict=return_dict,
                    output_hidden_states=True,
                )

            original_batch_size = origin["input_ids"].size(0)
            # Apply L2 loss on last hidden states
            original_hidden_states = combined_outputs.hidden_states[-1][:original_batch_size]
            augmented_hidden_states = combined_outputs.hidden_states[-1][original_batch_size:]

            unique_sample_idxs = augmented["sample_idxs"].unique()

            for idx in unique_sample_idxs:
                sample_mask = augmented["sample_idxs"] == idx
                current_augmented_hidden = augmented_hidden_states[sample_mask]
                current_augmented_labels = augmented['labels'][sample_mask]

                #augmented_cls_loss += augmented_outputs.loss
                augmented_answer_mask = current_augmented_labels != -100
                mask_expanded = augmented_answer_mask.unsqueeze(-1)
                augmented_answer_hidden = current_augmented_hidden.masked_select(mask_expanded).view(
                    current_augmented_hidden.size(0),
                    -1,
                    current_augmented_hidden.size(-1)
                )

                with torch.no_grad():
                    original_answer_mask = origin['labels'][idx] != -100
                    original_answer_hidden = original_hidden_states[idx, original_answer_mask, :]
                
                mse_loss = F.mse_loss(
                    original_answer_hidden.expand_as(augmented_answer_hidden), 
                    augmented_answer_hidden, 
                    reduction='mean'
                )
                consistency_loss += mse_loss

            consistency_loss = consistency_loss / len(unique_sample_idxs)

            if not torch.isnan(consistency_loss) and not torch.isinf(consistency_loss):
                total_loss = combined_outputs.loss + consistency_loss * self.consistency_loss_weight
            else:
                print("Warning: Invalid consistency loss detected, using only combined loss")
                total_loss = combined_outputs.loss

        else:
            combined_outputs = super().forward(
                input_ids=origin["input_ids"],
                pixel_values=origin["pixel_values"],
                attention_mask=origin["attention_mask"],
                labels=origin["labels"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss
            consistency_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self, 'training') and self.training:
            print(f"Total loss: {total_loss.item():.4f}, "
                  f"Original loss: {combined_outputs.loss.item():.4f}, "
                  f"Consistency loss: {consistency_loss.item():.4f}")
            
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
    
    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature
                


