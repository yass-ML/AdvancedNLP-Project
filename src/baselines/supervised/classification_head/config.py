"""
Configuration for Classification Head Fine-tuning.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ClassificationConfig:
    """Configuration for classification fine-tuning with a classification head."""

    # Model settings
    model_name: str
    num_labels: int  # Number of classification classes

    # Data settings
    dataset_path: str
    text_column: str = "problem"  # Column containing input text
    label_column: str = "type"    # Column containing labels

    # Training settings
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_seq_len: int = 512
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # LoRA settings (for PEFT)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization settings
    load_in_4bit: bool = False  # 4-bit quantization (QLoRA)
    load_in_8bit: bool = False  # 8-bit quantization

    # Output settings
    run_name: str = "classification_default"
    output_dir: str = "fine_tunings/classification_head"

    # Data subset for debugging
    subset_limit: Optional[float | int] = None

    # Focal loss settings for handling class imbalance
    use_focal_loss: bool = False
    focal_gamma: float = 2.0  # Focusing parameter (higher = more focus on hard examples)
    use_class_weights: bool = True  # Use class weights with focal loss
    class_weight_method: str = "effective"  # "inverse", "inverse_sqrt", or "effective"

    # Label mapping (will be populated during data loading)
    label2id: dict = field(default_factory=dict)
    id2label: dict = field(default_factory=dict)


def get_config(
    model_name: str = "google/gemma-2b",
    num_labels: int = 7,  # Will be overridden by actual dataset classes
    dataset_path: str = "qwedsacf/competition_math",
    text_column: str = "problem",
    label_column: str = "type",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_len: int = 512,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    run_name: str = "classification_default",
    output_dir: str = "fine_tunings/classification_head",
    subset_limit: Optional[float | int] = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_weights: bool = True,
    class_weight_method: str = "effective",
) -> ClassificationConfig:
    """Create a classification configuration with sensible defaults."""

    if target_modules is None:
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return ClassificationConfig(
        model_name=model_name,
        num_labels=num_labels,
        dataset_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        run_name=run_name,
        output_dir=output_dir,
        subset_limit=subset_limit,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_class_weights=use_class_weights,
        class_weight_method=class_weight_method,
    )


def load_config(config_path: str) -> ClassificationConfig:
    """Load a ClassificationConfig from a saved JSON file."""
    path = Path(config_path)

    # Try current directory and parents
    candidates = [path, path / "training_config.json"]
    if path.is_dir():
        candidates = [path / "training_config.json", path.parent / "training_config.json"]

    config_file = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            config_file = candidate
            break

    if config_file is None:
        raise FileNotFoundError(
            f"Could not find training_config.json in {config_path} or its parents."
        )

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    return ClassificationConfig(**config_dict)


def save_config(config: ClassificationConfig, output_dir: str) -> None:
    """Save configuration to JSON file."""
    from dataclasses import asdict

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
