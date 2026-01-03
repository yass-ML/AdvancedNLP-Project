from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    load_in_4bit: bool
    dataset_path: str
    batch_size: int
    learning_rate: float
    max_seq_len: int
    num_epochs: int
    lora_rank: int
    lora_alpha: int
    target_modules: list
    subset_limit: float | int | None = None

def get_config(
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
    dataset_path: str = "qwedsacf/competition_math",
    load_in_4bit: bool = True,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_len: int = 2048,
    num_epochs: int = 1,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    target_modules: list = None,
    subset_limit: float | int | None = None
) -> TrainingConfig:
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return TrainingConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        load_in_4bit=load_in_4bit,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        subset_limit=subset_limit
    )
