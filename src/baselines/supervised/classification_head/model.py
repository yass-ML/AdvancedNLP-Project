"""
Model loading for Classification Head Fine-tuning.

Uses AutoModelForSequenceClassification with optional LoRA/QLoRA.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from .config import ClassificationConfig


# Model registry mapping short names to HuggingFace model IDs
_MODEL_MAP = {
    # Gemma
    "gemma:2b": "google/gemma-2b",
    "gemma:7b": "google/gemma-7b",
    "gemma:2b-instruct": "google/gemma-2b-it",
    "gemma:7b-instruct": "google/gemma-7b-it",

    # Llama 3
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "llama3:8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",

    # Mistral
    "mistral:7b": "mistralai/Mistral-7B-v0.3",
    "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",

    # Phi-3
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",

    # Qwen 2 (accessible, no gating)
    "qwen2:0.5b": "Qwen/Qwen2-0.5B",
    "qwen2:1.5b": "Qwen/Qwen2-1.5B",
    "qwen2:7b": "Qwen/Qwen2-7B",
    "qwen2:7b-instruct": "Qwen/Qwen2-7B-Instruct",

    # Smaller models for testing
    "bert-base": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

# Target modules for different model architectures
_TARGET_MODULES_MAP = {
    # Decoder-only LLMs (Llama, Mistral, Gemma, Qwen, Phi)
    "llm": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # BERT-style encoder models
    "bert": ["query", "value", "key"],
    # DistilBERT
    "distilbert": ["q_lin", "v_lin", "k_lin"],
    # RoBERTa (same as BERT)
    "roberta": ["query", "value", "key"],
}


def _get_target_modules(model, default_modules: list) -> list:
    """
    Automatically determine the correct target modules based on model architecture.

    Args:
        model: The loaded model
        default_modules: Default modules from config

    Returns:
        list of target module names
    """
    model_type = getattr(model.config, "model_type", "").lower()

    # Check for known model types
    if "distilbert" in model_type:
        return _TARGET_MODULES_MAP["distilbert"]
    elif "roberta" in model_type:
        return _TARGET_MODULES_MAP["roberta"]
    elif "bert" in model_type:
        return _TARGET_MODULES_MAP["bert"]
    elif model_type in ["llama", "mistral", "gemma", "gemma2", "qwen2", "phi", "phi3"]:
        return _TARGET_MODULES_MAP["llm"]

    # Try to find linear layers in the model
    # This is a fallback for unknown architectures
    linear_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the last part of the module name
            parts = name.split(".")
            if parts:
                linear_module_names.add(parts[-1])

    # If we found linear layers, use them
    # Filter to common attention-related names
    attention_keywords = ["query", "key", "value", "q_", "k_", "v_", "proj", "lin"]
    filtered = [
        name for name in linear_module_names
        if any(kw in name.lower() for kw in attention_keywords)
    ]

    if filtered:
        print(f"Auto-detected target modules for {model_type}: {filtered}")
        return filtered

    # Fall back to default
    print(f"Using default target modules: {default_modules}")
    return default_modules


def get_model_id(model_name: str) -> str:
    """Resolve short model name to HuggingFace model ID."""
    return _MODEL_MAP.get(model_name, model_name)


def load_model(config: ClassificationConfig):
    """
    Load a model for sequence classification with optional LoRA.

    Args:
        config: ClassificationConfig with model settings

    Returns:
        tuple: (model, tokenizer)
    """
    model_id = get_model_id(config.model_name)

    # Configure quantization if requested
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure tokenizer has pad token (many LLMs don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=config.num_labels,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        device_map="auto",
        trust_remote_code=True,
        # Map labels for interpretability
        id2label=config.id2label if config.id2label else None,
        label2id=config.label2id if config.label2id else None,
    )

    # Configure padding side for classification
    # Left padding is common for causal LMs, but classification typically uses right padding
    tokenizer.padding_side = "right"

    # Ensure model config has pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if requested
    if config.use_lora:
        # Prepare model for k-bit training if quantized
        if config.load_in_4bit or config.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        # Determine target modules based on model architecture
        target_modules = _get_target_modules(model, config.target_modules)

        # Configure LoRA
        # modules_to_save ensures the classification head is trained (not frozen)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["score"],  # Explicitly train the classification head!
            bias="none",
            task_type=TaskType.SEQ_CLS,  # Sequence classification task
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_model_for_inference(model_path: str, config: ClassificationConfig = None):
    """
    Load a fine-tuned classification model for inference.

    Args:
        model_path: Path to the saved model (with LoRA adapters)
        config: Optional config, will be loaded from model_path if not provided

    Returns:
        tuple: (model, tokenizer)
    """
    from .config import load_config
    from peft import PeftModel

    if config is None:
        config = load_config(model_path)

    model_id = get_model_id(config.model_name)

    # Load tokenizer from fine-tuned path (might have special tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Ensure id2label uses integer keys (JSON loads them as strings)
    id2label = {int(k): v for k, v in config.id2label.items()} if config.id2label else None
    label2id = config.label2id  # label2id keys are strings, which is correct

    # Load base model (suppress warning about uninitialized classification head
    # since we'll load our trained weights immediately after)
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=config.num_labels,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        device_map="auto",
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id,
    )

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapters if they exist
    if config.use_lora:
        print(f"Loading LoRA adapters from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)

        # Only merge if NOT using quantization (creating 4-bit/8-bit merge issues)
        if not (config.load_in_4bit or config.load_in_8bit):
            model = model.merge_and_unload()  # Merge for faster inference
            print("LoRA adapters merged successfully.")
        else:
            print("Skipping LoRA merge (quantization active) to prevent precision loss.")
    else:
        model = base_model

    # Load the classification head (score layer) if saved separately
    import os
    classifier_head_path = os.path.join(model_path, "classifier_head.pt")
    if os.path.exists(classifier_head_path):
        print(f"Loading classification head from {classifier_head_path}...")
        classifier_state_dict = torch.load(classifier_head_path, map_location="cpu", weights_only=True)

        # Handle different saved formats (old PEFT-wrapped vs new clean format)
        # IMPORTANT: modules_to_save.default.weight contains the TRAINED weights
        # original_module.weight contains the ORIGINAL (untrained) weights
        if 'weight' in classifier_state_dict:
            # New clean format
            weight_tensor = classifier_state_dict['weight']
        elif 'modules_to_save.default.weight' in classifier_state_dict:
            # Old PEFT format - this is the TRAINED weight!
            weight_tensor = classifier_state_dict['modules_to_save.default.weight']
            print("  (using modules_to_save.default.weight - the trained weights)")
        elif 'original_module.weight' in classifier_state_dict:
            # Fallback to original (untrained) - should not normally happen
            print("  WARNING: Only original_module.weight found, these may be untrained!")
            weight_tensor = classifier_state_dict['original_module.weight']
        else:
            raise ValueError(f"Unknown classifier_head.pt format. Keys: {classifier_state_dict.keys()}")

        # Load into the model
        if hasattr(model, 'score'):
            model.score.weight.data.copy_(weight_tensor)
            print("Classification head (score) loaded successfully.")
        elif hasattr(model, 'classifier'):
            model.classifier.weight.data.copy_(weight_tensor)
            print("Classification head (classifier) loaded successfully.")
    else:
        print(f"WARNING: No classifier_head.pt found at {classifier_head_path}")
        print("The classification head weights may be randomly initialized!")

    model.eval()
    return model, tokenizer
