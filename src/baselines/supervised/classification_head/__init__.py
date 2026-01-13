"""
Classification Head Fine-tuning Module

This module implements proper sequence classification by adding a classification
head on top of pretrained language models, rather than using generative classification.

Benefits over generative classification:
- Single forward pass vs. autoregressive token generation
- Direct softmax over classes for more reliable predictions
- Faster inference and more stable training
- Standard classification metrics (cross-entropy loss)
"""

from .config import ClassificationConfig, get_config, load_config
from .model import load_model
from .data_loader import get_dataset, ClassificationDataCollator
from .trainer import train
from .evaluate import evaluate_model

__all__ = [
    "ClassificationConfig",
    "get_config",
    "load_config",
    "load_model",
    "get_dataset",
    "ClassificationDataCollator",
    "train",
    "evaluate_model",
]
