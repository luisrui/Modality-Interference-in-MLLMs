import argparse
import torch
import wandb
import os
import json

from torch.utils.data import DataLoader
from types import SimpleNamespace
from transformers import TrainingArguments
# from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from accelerate.utils import set_seed

from modules.models import PeftModelForLLaVAConsistency
from modules.utils import freeze_parameters, read_yaml, find_all_linear_names, get_model_and_processor, get_trainer
from modules.data import LlavaVQADataset, LLaVADataCollator, LlavaVQAOriginDataset, LLaVAOriginDataCollator, ModalityAnchoredBatchSampler

def train(args):
    """
    LLaVA fine-tuning script
    """

    ds_config = json.load(open(args.deepspeed_config))

    if args.seed:
        set_seed(args.seed)
        
    if args.local_rank == 0:
        wandb_run_id = args.wandb_run_id if args.wandb_run_id else None
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            entity=args.wandb_entity,
            id=wandb_run_id, 
            resume="auto",
        )
        os.makedirs(args.save_dir, exist_ok=True)
        args_path = os.path.join(args.save_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # 1. Load processor, model
    model, processor = get_model_and_processor(args)
    if hasattr(model, 'set_con_loss_weight_and_tem'):
        model.set_con_loss_weight_and_tem(args.consistency_loss_weight, args.temperature)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    model.enable_input_require_grads()
    if 'lora' in args.finetune_type:
        lora_config = LoraConfig(
            r=args.lora_r,  
            lora_alpha=args.lora_alpha, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM", 
            inference_mode=False 
        )
        if args.consistency_type:
            model = PeftModelForLLaVAConsistency(
                model,
                lora_config,
                adapter_name='default',
                autocast_adapter_dtype=True,
                low_cpu_mem_usage=False,
            )
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif 'full' in args.finetune_type:
        # freeze_parameters(model.vision_tower, model.language_model) # for debug only
        freeze_parameters(model.vision_tower)

    # 2. Load data
    if args.consistency_type:
        train_data = LlavaVQADataset(args, processor, mode="train")
        test_data = LlavaVQADataset(args, processor, mode="val")
    else:
        train_data = LlavaVQAOriginDataset(args, processor, mode="train")
        test_data = LlavaVQAOriginDataset(args, processor, mode="val")

    image_per_batch = int(args.per_device_batch_size * args.image_heavy_ratio)
    text_per_batch = int(args.per_device_batch_size * args.text_heavy_ratio)
    vqa_per_batch = args.per_device_batch_size - image_per_batch - text_per_batch
    modality_sampler = ModalityAnchoredBatchSampler(
        text_idxs=train_data.text_idxs,
        image_idxs=train_data.image_idxs,
        vqa_idxs=train_data.vqa_idxs,
        image_per_batch=image_per_batch,
        text_per_batch=text_per_batch,
        vqa_per_batch=vqa_per_batch,
    )
    
    if not args.consistency_type:
        data_collator = LLaVAOriginDataCollator(processor)
    else:
        data_collator = LLaVADataCollator(processor)

    train_loader = DataLoader(
        train_data,
        batch_sampler=modality_sampler,
        collate_fn=data_collator,   
        num_workers=24,
        pin_memory=True,
    )

    if args.local_rank == 0 or args.local_rank == -1:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params}")
        print(f"Total params: {total_params}")

    # 4. Define train hyperparams
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        output_dir=args.save_dir,
        save_strategy="steps",
        save_steps=args.save_eval_steps,
        save_total_limit=args.save_total_limit,
        ddp_find_unused_parameters=True,
        tf32=args.tf32, # mixed precision
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=ds_config
    )
    
    # 5. Define trainer
    trainer = get_trainer(
            args=args, 
            model=model, 
            training_args=training_args, 
            train_data=train_data, 
            test_data=test_data, 
            metrics=None,
            data_collator=data_collator,
            train_loader=train_loader
        )

    # 6. Train
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
         trainer.train()

    if "lora" in args.finetune_type:
        if args.local_rank == 0 or args.local_rank == -1:
            trainer.save_state()
            model.save_pretrained(args.save_dir)
            processor.save_pretrained(args.save_dir)

def main(args):
    train(args)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    parse.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parse.parse_args()
    yaml_args = read_yaml(path=args.config)
    yaml_args.update(vars(args))
    args = SimpleNamespace(**yaml_args)
    main(args)