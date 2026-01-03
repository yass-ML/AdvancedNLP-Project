from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from .model import load_model
from .data_loader import get_dataset
from .config import TrainingConfig

def train(config: TrainingConfig):
    # 1. Load Model (Unsloth Wrapper) & Tokenizer
    model, tokenizer = load_model(config)

    # 2. Load Data (Formatted)
    train_dataset, test_dataset = get_dataset(config)

    # 3. Training Arguments (The Control Panel)
    args = TrainingArguments(
        output_dir = "outputs",
        per_device_train_batch_size = config.batch_size,
        gradient_accumulation_steps = 4, # Simulate larger batches
        warmup_steps = 5,                # Gently ramp up learning rate
        max_steps = 60,                  # Short run for testing (use num_train_epochs usually)
        # num_train_epochs = config.num_epochs,
        learning_rate = config.learning_rate,

        # Memory Optimizations
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(), # Use BF16 if available (RTX 3060 yes!)
        optim = "adamw_8bit",           # Saves huge memory
        weight_decay = 0.01,

        logging_steps = 1,
        seed = 3407,
    )

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_len,
        dataset_num_proc = 2,
        args = args,
    )

    # 5. Train & Save
    print("Starting training...")
    trainer_stats = trainer.train()

    print("Saving model adapters...")
    # This saves ONLY the LoRA adapters (small file), not the whole model
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

    return trainer_stats
