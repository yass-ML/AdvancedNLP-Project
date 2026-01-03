try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None


from .config import TrainingConfig


def load_model(config : TrainingConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
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
