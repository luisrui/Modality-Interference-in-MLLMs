import wandb
import torch
import deepspeed
from torch import nn as nn

from typing import Dict

from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import SaveStrategy

class BatchSampleTrainer(Trainer):
    '''
    A Trainer that allows for different sampling ratio of different datasets in one batch.
    '''
    def __init__(self, *args, train_loader=None, **kwargs):
        self.custom_train_dataloader = train_loader  
        kwargs.pop("train_loader", None)           
        super().__init__(*args, **kwargs)  

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader
        return super().get_train_dataloader()  
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        if "data_types" in inputs:
            data_types = inputs["data_types"]
            del inputs["data_types"]

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
class ConsistencyTrainer(BatchSampleTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Record the loss to wandb. (total loss, original next token prediction loss, consistency loss)
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        if "data_types" in inputs:
            data_types = inputs["data_types"]
            del inputs["data_types"]
            
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model._get_name()
            
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        original_loss = outputs.get("original_loss", None)
        consistency_loss = outputs.get("consistency_loss", None)

        self.latest_losses = {
            "original_loss": original_loss.item() if original_loss is not None else None,
            "consistency_loss": consistency_loss.item() if consistency_loss is not None else None
        }

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
                       
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if hasattr(self, "latest_losses"):
                avg_original_loss = None
                avg_consistency_loss = None
                
                if self.latest_losses["original_loss"] is not None:
                    avg_original_loss = self._nested_gather(torch.tensor(self.latest_losses["original_loss"], device=self.model.device)).mean().item()
                if self.latest_losses["consistency_loss"] is not None:
                    avg_consistency_loss = self._nested_gather(torch.tensor(self.latest_losses["consistency_loss"], device=self.model.device)).mean().item()

                if avg_original_loss is not None:
                    logs["original_loss"] = avg_original_loss
                if avg_consistency_loss is not None:
                    logs["consistency_loss"] = avg_consistency_loss 
            
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
class RGPerturbationTrainer(BatchSampleTrainer):
    def __init__(self, *args, sigma, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def training_step(
        self, model: nn.Module, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step with RG Perturbation (Random Gaussian Noise).

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        torch._dynamo.reset()
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs) 
        if "data_types" in inputs:
            data_types = inputs["data_types"]
            del inputs["data_types"]
        #  Step 1: Standard Forward Pass
        with self.compute_loss_context_manager():
            standard_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        with self.compute_loss_context_manager():
            perturbed_loss = self.compute_perturb_loss(model, data_types, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            standard_loss = standard_loss.mean()
            perturbed_loss = perturbed_loss.mean()

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            standard_loss = standard_loss / self.args.gradient_accumulation_steps
            perturbed_loss = perturbed_loss / self.args.gradient_accumulation_steps
        
        self.log({"standard_loss": standard_loss.detach().item(), "perturbed_loss": perturbed_loss.detach().item()})

        total_loss = standard_loss + perturbed_loss
        self.accelerator.backward(total_loss)
        return total_loss.detach()
    
    def compute_perturb_loss(self, model, data_types, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the perturbation loss with new defined loss logic. (similar as model's original loss logic, but with non-gradient perturbations)
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model.ADperturbed_forward(self.sigma, data_types, **inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

class PGDPerturbationTrainer(BatchSampleTrainer):
    def __init__(self, alpha, epsilon, pgd_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps

    def training_step(
        self, model: nn.Module, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step with RG Perturbation (Random Gaussian Noise).

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs) 
        if "data_types" in inputs:
            data_types = inputs["data_types"]
            del inputs["data_types"]
        #  Step 1: Standard Forward Pass
        # with self.compute_loss_context_manager():
        #     standard_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # Doing standard forward and perturbed forward in one step
        with self.compute_loss_context_manager():
            total_loss, outputs = self.compute_perturb_loss(model, data_types, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            standard_loss = outputs.origin_loss
            perturbed_loss = outputs.perturbed_loss

        if self.args.n_gpu > 1:
            standard_loss = standard_loss.mean()
            perturbed_loss = perturbed_loss.mean()

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            standard_loss = standard_loss / self.args.gradient_accumulation_steps
            perturbed_loss = perturbed_loss / self.args.gradient_accumulation_steps
        
        self.log({"standard_loss": standard_loss.detach().item(), "perturbed_loss": perturbed_loss.detach().item()})

        self.accelerator.backward(total_loss)
        return total_loss.detach()
    
    def compute_perturb_loss(self, model, data_types, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the perturbation loss with new defined loss logic. (similar as model's original loss logic, but with non-gradient perturbations)
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # Forward with PGD
        outputs = model.ADperturbed_forward(self.alpha, self.epsilon, self.pgd_steps, data_types, **inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

class ConsistencyPGDPerturbationTrainer(PGDPerturbationTrainer):
    def __init__(self, alpha, epsilon, pgd_steps, *args, **kwargs):
        # Should call the PGDPerturbationTrainer's __init__ method first
        super().__init__(alpha, epsilon, pgd_steps, *args, **kwargs)
    
    def compute_loss(self, model, data_types, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the origin loss and perturbed loss with new defined loss logic. (similar as model's original loss logic, but with non-gradient perturbations)
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model.forward(self.alpha, self.epsilon, self.pgd_steps, data_types, **inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        original_loss = outputs.get("original_loss", None)
        perturbed_loss = outputs.get("perturbed_loss", None)
        consistency_loss = outputs.get("consistency_loss", None)

        self.latest_losses = {
            "original_loss": original_loss.item() if original_loss is not None else None,
            "perturbed_loss": perturbed_loss.item() if perturbed_loss is not None else None,
            "consistency_loss": consistency_loss.item() if consistency_loss is not None else None
        }
        
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    def training_step(
        self, model: nn.Module, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs) 
        if "data_types" in inputs:
            data_types = inputs["data_types"]
            del inputs["data_types"]
        #  Step 1: Standard Forward Pass
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, data_types, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.log({"loss": loss.detach().item()})

        self.accelerator.backward(loss)
        return loss.detach()
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
                       
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if hasattr(self, "latest_losses"):
                avg_original_loss = None
                avg_consistency_loss = None
                
                if self.latest_losses["original_loss"] is not None:
                    avg_original_loss = self._nested_gather(torch.tensor(self.latest_losses["original_loss"], device=self.model.device)).mean().item()
                    logs["original_loss"] = avg_original_loss
                if self.latest_losses["consistency_loss"] is not None:
                    avg_consistency_loss = self._nested_gather(torch.tensor(self.latest_losses["consistency_loss"], device=self.model.device)).mean().item()
                    logs["consistency_loss"] = avg_consistency_loss 
                if self.latest_losses["perturbed_loss"] is not None:
                    avg_perturbed_loss = self._nested_gather(torch.tensor(self.latest_losses["perturbed_loss"], device=self.model.device)).mean().item()
                    logs["perturbed_loss"] = avg_perturbed_loss

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)