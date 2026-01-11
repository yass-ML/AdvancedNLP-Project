import json
import os
from dataclasses import asdict

import torch

from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

from .model import load_model
from .data_loader import get_dataset
from .config import TrainingConfig


def train(config: TrainingConfig):
    # 1. Load Model (Unsloth Wrapper) & Tokenizer
    model, tokenizer = load_model(config)

    # 2. Load Data (Formatted)
    train_dataset, val_dataset, test_dataset = get_dataset(config)

    # 3. Training Arguments (The Control Panel)
    checkpoint_dir = f"fine_tunings/{config.run_name}/checkpoints"
    args = TrainingArguments(
        output_dir = checkpoint_dir,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = 1,  # Reduce to avoid OOM during validation
        gradient_accumulation_steps = 4, # Simulate larger batches
        warmup_steps = 5,                # Gently ramp up learning rate
        # max_steps = 60,                # Short run for testing
        num_train_epochs = config.num_epochs,
        learning_rate = config.learning_rate,

        # Memory Optimizations
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(), # Use BF16 if available
        optim = "adamw_8bit",           # Saves huge memory
        weight_decay = 0.01,

        logging_steps = 1,

        # Evaluation Strategy
        eval_strategy = "steps",         # Evaluate during training
        eval_steps = 500,                # Evaluate every 500 steps (approx 9 times total)
        save_strategy = "steps",         # Must match eval_strategy
        save_steps = 500,
        load_best_model_at_end = True,   # Load best model when finished
        metric_for_best_model = "loss",  # Monitor validation loss

        # TensorBoard logging
        report_to = "tensorboard",
        logging_dir = f"fine_tunings/{config.run_name}/logs",

        seed = 3407,
    )

    # 4. Initialize Trainer (Standard)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,      # Pass validation set
        dataset_text_field = "text",
        max_seq_length = config.max_seq_len,
        dataset_num_proc = 2,
        args = args,
    )

    # 5. Train & Save
    print("Starting training...")
    trainer_stats = trainer.train()

    # Create output directory
    output_dir = f"fine_tunings/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving BEST model adapters (load_best_model_at_end=True) to {output_dir}...")
    # This saves ONLY the LoRA adapters (small file), not the whole model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config for reproducibility
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Training config saved to {output_dir}/training_config.json")

    return trainer_stats
