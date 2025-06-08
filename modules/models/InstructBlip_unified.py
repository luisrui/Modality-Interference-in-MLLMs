import torch

from transformers.utils.generic import ModelOutput

from .InstructBlip_consistency_regularization import *
from .InstructBlip_ad_attacks import *

@dataclass
class InstructBlipOutputWithConsistencyLossAndAdversarial(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    vision_outputs: Optional[Any] = None
    qformer_outputs: Optional[Any] = None
    language_model_outputs: Optional[Any] = None
    consistency_loss: Optional[torch.FloatTensor] = None
    original_loss: Optional[torch.FloatTensor] = None
    perturbed_loss: Optional[torch.FloatTensor] = None

class InstructBlipForKLConsistencyPGDPerturbation(InstructBlipForPGDPerturbation, InstructBlipForKLdivergence):
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
    ) -> Union[Tuple, InstructBlipOutputWithConsistencyLossAndAdversarial]:

        consistency_loss = torch.tensor(0.0, device=self.device)

        if augmented is not None:
            original_batch_size = origin["input_ids"].size(0)

            combined_inputs = {
                "input_ids": torch.cat([origin["input_ids"], augmented["input_ids"]], dim=0),
                "attention_mask": torch.cat([origin["attention_mask"], augmented["attention_mask"]], dim=0),
                "pixel_values": torch.cat([origin["pixel_values"], augmented["pixel_values"]], dim=0),
                "qformer_input_ids": torch.cat([origin["qformer_input_ids"], augmented["qformer_input_ids"]], dim=0),
                "qformer_attention_mask": torch.cat([origin["qformer_attention_mask"], augmented["qformer_attention_mask"]], dim=0),
                "labels": torch.cat([origin["labels"], augmented["labels"]], dim=0),
            }

            # PGD + forward
            combined_outputs = super().ADperturbed_forward(
                alpha=alpha,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                data_types=data_types,
                pixel_values=combined_inputs["pixel_values"],
                qformer_input_ids=combined_inputs["qformer_input_ids"],
                input_ids=combined_inputs["input_ids"],
                attention_mask=combined_inputs["attention_mask"],
                labels=combined_inputs["labels"],
                qformer_attention_mask=combined_inputs["qformer_attention_mask"],
                return_dict=return_dict,
            )

            original_logits = combined_outputs.logits[:original_batch_size]
            augmented_logits = combined_outputs.logits[original_batch_size:]

            matched_sample_idxs = torch.cat([
                augmented["sample_idxs"], origin["sample_idxs"], augmented["sample_idxs"]
            ], dim=0)
            expanded_labels = torch.cat([
                augmented["labels"], origin["labels"], augmented["labels"]
            ], dim=0)
            unique_sample_idxs = matched_sample_idxs.unique()

            for idx in unique_sample_idxs:
                sample_mask = matched_sample_idxs == idx
                current_augmented_logits = augmented_logits[sample_mask]
                current_augmented_labels = expanded_labels[sample_mask]

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
                    original_answer_logits = original_logits[idx, original_answer_mask, :]

                q_prob = F.softmax(augmented_answer_logits / self.temp, dim=-1)
                q_prob = q_prob.clamp(min=1e-6)
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
                pixel_values=origin["pixel_values"],
                qformer_input_ids=origin["qformer_input_ids"],
                input_ids=origin["input_ids"],
                attention_mask=origin["attention_mask"],
                labels=origin["labels"],
                qformer_attention_mask=origin["qformer_attention_mask"],
                return_dict=return_dict,
            )
            total_loss = combined_outputs.loss

        return InstructBlipOutputWithConsistencyLossAndAdversarial(
            loss=total_loss,
            logits=combined_outputs.logits,
            vision_outputs=combined_outputs.vision_outputs,
            qformer_outputs=combined_outputs.qformer_outputs,
            language_model_outputs=combined_outputs.language_model_outputs,
            consistency_loss=consistency_loss,
            original_loss=combined_outputs.origin_loss,
            perturbed_loss=combined_outputs.perturbed_loss,
        )

    def set_con_loss_weight_and_tem(self, weight, temperature):
        self.consistency_loss_weight = weight
        self.temp = temperature
