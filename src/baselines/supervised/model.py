try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None


from .config import TrainingConfig
from .model_registry import get_model_id


def load_model(config : TrainingConfig):
    # Resolve short model name to full HuggingFace ID
    model_id = get_model_id(config.model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = config.max_seq_len,
        load_in_4bit = config.load_in_4bit,
        dtype = None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config.lora_rank,
        target_modules = config.target_modules,
        lora_alpha = config.lora_alpha,
        lora_dropout = 0, # Unlsoth requires 0 for optimization
        bias = "none",
        use_gradient_checkpointing = "unsloth"
    )

    return model, tokenizer
