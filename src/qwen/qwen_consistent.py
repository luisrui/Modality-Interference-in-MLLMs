import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import json
import wandb

from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from modules.utils import freeze_parameters, read_yaml, get_model_and_processor, get_trainer
from modules.data import QwenVQAOriginDataset, QwenVQADataset, QwenDataCollator, QwenOriginDataCollator, ModalityAnchoredBatchSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def train(args):
    """
    Qwen fine-tuning script
    """
    
    ds_config = json.load(open(args.deepspeed_config))
    ds_config["bf16"] = {"enabled": args.bf16}
    ds_config["fp16"] = {"enabled": args.fp16}

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

    # Load processor, model
    model, processor = get_model_and_processor(args)
    if hasattr(model, 'set_con_loss_weight_and_tem'):
        model.set_con_loss_weight_and_tem(args.consistency_loss_weight, args.temperature)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    
    model.enable_input_require_grads()
    if 'full' in args.finetune_type:
        freeze_parameters(model.visual, exclude_names=["merger"]) ## freeze visual encoder, but unfreeze visual.merger
        # pass # Try to train all parameters

    if args.consistency_type:
        train_data = QwenVQADataset(args, processor, mode="train")
        test_data = QwenVQADataset(args, processor, mode="val")
    else:
        train_data = QwenVQAOriginDataset(args, processor, mode="train")
        test_data = QwenVQAOriginDataset(args, processor, mode="val")

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
        data_collator = QwenOriginDataCollator(processor, args.max_length)
    else:
        data_collator = QwenDataCollator(processor, args.max_length)

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
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
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

def main(args):
    train(args)

if __name__ == '__main__':
    #args = parse_args()
    parse = argparse.ArgumentParser()
    parse.add_argument("--local_rank", type=int, default=-1) 
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")

    args = parse.parse_args()
    yaml_args = read_yaml(path=args.config)
    yaml_args.update(vars(args))
    args = SimpleNamespace(**yaml_args)
    main(args)