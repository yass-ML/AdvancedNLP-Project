"""
Data loading for Classification Head Fine-tuning.

Prepares tokenized datasets with integer labels for sequence classification.
"""

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch

from .config import ClassificationConfig


@dataclass
class ClassificationDataCollator:
    """
    Data collator for sequence classification.
    Handles padding and batching of tokenized examples.
    """
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels from features
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None

        # Pad the inputs
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Add labels back
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def create_label_mappings(dataset: Dataset, label_column: str) -> tuple[dict, dict]:
    """
    Create label2id and id2label mappings from the dataset.

    Args:
        dataset: HuggingFace dataset
        label_column: Name of the column containing labels

    Returns:
        tuple: (label2id dict, id2label dict)
    """
    # Get unique labels
    unique_labels = sorted(list(set(dataset[label_column])))

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    return label2id, id2label


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: ClassificationConfig,
    label2id: dict,
) -> Dataset:
    """
    Tokenize dataset for sequence classification.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use
        config: Classification configuration
        label2id: Mapping from label strings to integers

    Returns:
        Tokenized dataset with 'input_ids', 'attention_mask', and 'labels'
    """

    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples[config.text_column],
            truncation=True,
            max_length=config.max_seq_len,
            padding=False,  # We'll pad in the data collator
        )

        # Convert string labels to integers
        tokenized["labels"] = [label2id[label] for label in examples[config.label_column]]

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def get_dataset(config: ClassificationConfig, tokenizer: PreTrainedTokenizer):
    """
    Load, split, and tokenize the dataset for classification.

    Args:
        config: Classification configuration
        tokenizer: Tokenizer to use

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, label2id, id2label)
    """
    # 1. Load the dataset
    print(f"Loading dataset: {config.dataset_path}")
    dataset = load_dataset(config.dataset_path, split="train")

    # 2. Filter out None values
    dataset = dataset.filter(
        lambda x: x[config.label_column] is not None and x[config.text_column] is not None
    )

    # 3. Optional: Fast Debugging Subset
    if config.subset_limit:
        if isinstance(config.subset_limit, float):
            limit = int(len(dataset) * config.subset_limit)
        else:
            limit = config.subset_limit
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Using subset of {len(dataset)} samples")

    # 4. Create label mappings BEFORE splitting
    label2id, id2label = create_label_mappings(dataset, config.label_column)
    num_labels = len(label2id)
    print(f"Found {num_labels} classes: {sorted(label2id.keys())}")

    # 5. Split into Train (80%), Val (10%), Test (10%)
    # First split off 10% for test
    main_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = main_split["test"]
    remaining_dataset = main_split["train"]

    # Split remaining into Train and Val
    train_val_split = remaining_dataset.train_test_split(test_size=0.1/0.9, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 6. Tokenize all splits
    train_dataset = tokenize_dataset(train_dataset, tokenizer, config, label2id)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, config, label2id)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, config, label2id)

    return train_dataset, val_dataset, test_dataset, label2id, id2label


def get_label_info(config: ClassificationConfig) -> tuple[int, dict, dict]:
    """
    Get label information from the dataset without full loading/tokenization.

    This is useful for initializing the model with the correct number of labels
    before loading the full dataset.

    Args:
        config: Classification configuration

    Returns:
        tuple: (num_labels, label2id, id2label)
    """
    # Load dataset
    dataset = load_dataset(config.dataset_path, split="train")

    # Filter None values
    dataset = dataset.filter(
        lambda x: x[config.label_column] is not None
    )

    # Optional subset (for consistency with get_dataset)
    if config.subset_limit:
        if isinstance(config.subset_limit, float):
            limit = int(len(dataset) * config.subset_limit)
        else:
            limit = config.subset_limit
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Create label mappings
    label2id, id2label = create_label_mappings(dataset, config.label_column)

    return len(label2id), label2id, id2label


def get_raw_dataset(config: ClassificationConfig):
    """
    Load the raw dataset without tokenization.
    Useful for inspection and analysis.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, label2id, id2label)
    """
    # Load dataset
    dataset = load_dataset(config.dataset_path, split="train")

    # Filter None values
    dataset = dataset.filter(
        lambda x: x[config.label_column] is not None and x[config.text_column] is not None
    )

    # Optional subset
    if config.subset_limit:
        if isinstance(config.subset_limit, float):
            limit = int(len(dataset) * config.subset_limit)
        else:
            limit = config.subset_limit
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Create label mappings
    label2id, id2label = create_label_mappings(dataset, config.label_column)

    # Split
    main_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = main_split["test"]
    remaining_dataset = main_split["train"]

    train_val_split = remaining_dataset.train_test_split(test_size=0.1/0.9, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    return train_dataset, val_dataset, test_dataset, label2id, id2label
