"""
Training loop for Classification Head Fine-tuning.

Uses HuggingFace Trainer with classification-specific metrics.
"""

import os
import json
import numpy as np
from dataclasses import asdict
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from .config import ClassificationConfig, save_config
from .model import load_model
from .data_loader import get_dataset, ClassificationDataCollator


def compute_metrics(eval_pred):
    """
    Compute classification metrics for the Trainer.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids

    Returns:
        dict with accuracy, precision, recall, f1
    """
    predictions, labels = eval_pred

    # Get predicted class (argmax of logits)
    preds = np.argmax(predictions, axis=1)

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class FocalLossTrainer(Trainer):
    """Custom Trainer that uses Focal Loss for handling class imbalance."""

    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to use focal loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.focal_loss_fn is not None:
            loss = self.focal_loss_fn(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train(
    config: ClassificationConfig,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Train a classification model with a classification head.

    Args:
        config: Classification configuration
        resume_from_checkpoint: Optional path to checkpoint to resume from

    Returns:
        TrainOutput with training results
    """
    from .data_loader import get_label_info
    from .losses import FocalLoss, compute_class_weights

    # 1. First, get label information from dataset to configure model correctly
    print("Analyzing dataset for label information...")
    num_labels, label2id, id2label = get_label_info(config)

    # Update config with label mappings BEFORE loading model
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    print(f"Found {num_labels} classes: {sorted(label2id.keys())}")

    # 2. Load model and tokenizer with correct number of labels
    print(f"Loading model: {config.model_name}")
    model, tokenizer = load_model(config)

    # 3. Load and tokenize data
    print("Loading and tokenizing dataset...")
    train_dataset, val_dataset, test_dataset, label2id, id2label = get_dataset(
        config, tokenizer
    )

    # 4. Set up focal loss if enabled
    focal_loss_fn = None
    if config.use_focal_loss:
        print(f"Using Focal Loss with gamma={config.focal_gamma}")

        # Compute class weights if enabled
        class_weights = None
        if config.use_class_weights:
            # Get labels from training set
            train_labels = train_dataset["labels"]
            class_weights = compute_class_weights(
                train_labels,
                num_labels,
                method=config.class_weight_method
            )
            print(f"Class weights ({config.class_weight_method}): {class_weights.tolist()}")

        focal_loss_fn = FocalLoss(
            alpha=class_weights,
            gamma=config.focal_gamma,
            reduction="mean"
        )

    # 5. Create data collator
    data_collator = ClassificationDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.max_seq_len,
    )

    # 6. Set up output directories
    output_dir = f"{config.output_dir}/{config.run_name}"
    checkpoint_dir = f"{output_dir}/checkpoints"
    logging_dir = f"{output_dir}/logs"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # 7. Configure training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,

        # Training hyperparameters
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,

        # Precision
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),

        # Optimizer
        optim="adamw_torch",

        # Evaluation strategy
        eval_strategy="steps",
        eval_steps=100,  # Evaluate frequently for early stopping
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Monitor validation loss
        greater_is_better=False,  # Lower loss is better

        # Logging
        logging_dir=logging_dir,
        logging_steps=10,
        report_to=["tensorboard"],

        # Misc
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # Keep all columns for data collator
    )

    # 8. Initialize trainer (use FocalLossTrainer if focal loss is enabled)
    TrainerClass = FocalLossTrainer if config.use_focal_loss else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    if config.use_focal_loss:
        trainer_kwargs["focal_loss_fn"] = focal_loss_fn

    trainer = TrainerClass(**trainer_kwargs)

    # 9. Train
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 10. Save final model
    print(f"Saving model to {output_dir}...")

    # Save the model (LoRA adapters if using PEFT, full model otherwise)
    if config.use_lora:
        # Save LoRA adapters
        model.save_pretrained(output_dir)

        # IMPORTANT: Also save the classification head separately
        # LoRA only saves adapter weights, not the classifier head (score layer)
        classifier_head_path = f"{output_dir}/classifier_head.pt"
        # Get the base model's classifier (score) layer
        base_model = model.get_base_model()

        # Extract the actual weight from potentially PEFT-wrapped module
        def get_classifier_weight(layer):
            """Extract the actual TRAINED weight tensor from a potentially PEFT-wrapped layer."""
            # IMPORTANT: Check modules_to_save FIRST - this contains the TRAINED weights!
            # original_module contains the ORIGINAL (untrained) weights
            if hasattr(layer, 'modules_to_save') and 'default' in layer.modules_to_save:
                # PEFT ModulesToSaveWrapper - get the TRAINED weights
                print(f"  Extracting from modules_to_save['default'] (trained weights)")
                return {'weight': layer.modules_to_save['default'].weight.data.clone()}
            elif hasattr(layer, 'original_module'):
                # Fallback - but this is likely untrained!
                print(f"  WARNING: Extracting from original_module (may be untrained!)")
                return {'weight': layer.original_module.weight.data.clone()}
            else:
                # Regular module (non-PEFT)
                return {'weight': layer.weight.data.clone()}

        if hasattr(base_model, 'score'):
            classifier_state = get_classifier_weight(base_model.score)
            torch.save(classifier_state, classifier_head_path)
            print(f"Saved classification head to {classifier_head_path}")
        elif hasattr(base_model, 'classifier'):
            classifier_state = get_classifier_weight(base_model.classifier)
            torch.save(classifier_state, classifier_head_path)
            print(f"Saved classification head to {classifier_head_path}")
    else:
        trainer.save_model(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save training config
    save_config(config, output_dir)

    # 11. Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)

    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(f"Test Results: {test_results}")

    # Save test results
    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Generate detailed classification report
    print("\nGenerating detailed classification report...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
    )
    print("\nClassification Report:")
    print(report)

    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nTraining complete! Model saved to {output_dir}")

    return train_result
