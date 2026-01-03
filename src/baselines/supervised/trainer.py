import json
import os
from collections import Counter
from dataclasses import asdict

import torch
from torch.utils.data import WeightedRandomSampler

from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

from .model import load_model
from .data_loader import get_dataset
from .config import TrainingConfig


def compute_sample_weights(dataset, label_column: str = "type"):
    """
    Compute per-sample weights for WeightedRandomSampler.
    Each sample's weight = 1 / (count of its class), so minority classes
    are sampled more frequently.
    """
    labels = dataset[label_column]
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Log class distribution
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION (Weighted Sampling)")
    print(f"{'='*60}")
    print(f"Total training samples: {total_samples}")
    print(f"Number of classes: {len(class_counts)}")

    # Sort by count (ascending) to highlight imbalance
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    max_count = sorted_counts[-1][1]
    min_count = sorted_counts[0][1]

    print(f"\nMajority class: '{sorted_counts[-1][0]}' ({max_count} samples)")
    print(f"Minority class: '{sorted_counts[0][0]}' ({min_count} samples)")
    print(f"Imbalance ratio: {max_count / min_count:.2f}x")

    print(f"\n{'Class':<25} {'Count':>8} {'Weight':>10} {'Effective':>12}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*12}")

    for class_name, count in sorted_counts:
        weight = 1.0 / count
        # Effective multiplier = how many times more likely to be sampled vs majority class
        effective_mult = max_count / count
        print(f"{class_name:<25} {count:>8} {weight:>10.6f} {effective_mult:>10.2f}x")

    print(f"{'='*60}\n")

    # Weight = 1/count so rarer classes have higher weight
    weights = [1.0 / class_counts[label] for label in labels]
    return torch.tensor(weights, dtype=torch.float)


class WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer with weighted sampling for class imbalance."""

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self, dataset=None):
        if self.sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return super()._get_train_sampler(dataset)

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
        bf16 = is_bfloat16_supported(), # Use BF16 if available (RTX 3060 yes!)
        optim = "adamw_8bit",           # Saves huge memory
        weight_decay = 0.01,

        logging_steps = 1,

        # Evaluation & Early Stopping Strategy
        eval_strategy = "steps",         # Evaluate during training
        eval_steps = 100,                # Evaluate every 100 steps
        save_strategy = "steps",         # Must match eval_strategy
        save_steps = 100,
        load_best_model_at_end = True,   # Load best model when finished
        metric_for_best_model = "loss",  # Monitor validation loss

        # TensorBoard logging
        report_to = "tensorboard",
        logging_dir = f"fine_tunings/{config.run_name}/logs",

        seed = 3407,
    )

    # 4. Compute sample weights for class balancing (memory-efficient)
    print("Computing sample weights for class balancing...")
    sample_weights = compute_sample_weights(train_dataset, label_column="type")

    # 5. Initialize Trainer with weighted sampling
    trainer = WeightedSFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,      # Pass validation set
        dataset_text_field = "text",
        max_seq_length = config.max_seq_len,
        dataset_num_proc = 2,
        args = args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        sample_weights = sample_weights  # Pass weights for balanced sampling
    )

    # 5. Train & Save
    print("Starting training...")
    trainer_stats = trainer.train()

    # Create output directory
    output_dir = f"fine_tunings/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving model adapters to {output_dir}...")
    # This saves ONLY the LoRA adapters (small file), not the whole model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config for reproducibility
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Training config saved to {output_dir}/training_config.json")

    return trainer_stats
