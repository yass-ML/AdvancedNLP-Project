from datasets import load_dataset
from .config import TrainingConfig

# This data loader works only for the math dataset for now

def make_formatting_func(valid_classes: list[str]):
    """
    Creates a formatting function with the valid classes baked in.
    """
    classes_str = ", ".join(sorted(valid_classes))

    def formatting_prompts_func(examples):
        """
        Formats the raw dataset into an instruction-tuning format.
        Unsloth/TRL will automatically train on the 'text' field.
        """
        instruction = f"Classify the following math problem into exactly one of the following categories: {classes_str}. Return only the category name. Do not add any additional text or explanation."

        problems = examples["problem"]
        types    = examples["type"]
        texts    = []
        prompts  = []

        for problem, type_ in zip(problems, types):
            # The standard Alpaca/Instruction format
            prompt = f"Instruction: {instruction}\nProblem: {problem}\nCategory: "
            text   = f"{prompt}{type_}"

            texts.append(text)
            prompts.append(prompt)

        return { "text" : texts, "prompt" : prompts }

    return formatting_prompts_func

def get_dataset(config:TrainingConfig):
    """
    Loads, splits, and formats the dataset.
    Returns: (train_dataset, test_dataset)
    """
    # 1. Load the dataset (only has 'train' split)
    dataset = load_dataset(config.dataset_path, split="train")

    # 2. Simple sanity check: remove None values
    dataset = dataset.filter(lambda x: x["type"] is not None and x["problem"] is not None)

    # 2.5 Optional: Fast Debugging Subset
    if config.subset_limit:
        if isinstance(config.subset_limit, float):
            limit = int(len(dataset) * config.subset_limit)
        else:
            limit = config.subset_limit
        dataset = dataset.select(range(limit))

    # 3. Extract unique classes BEFORE splitting (so both splits use same classes)
    valid_classes = list(set(dataset["type"]))
    print(f"Found {len(valid_classes)} classes: {sorted(valid_classes)}")

    # 4. Split into Train (80%), Val (10%), Test (10%)
    # First split off 10% for test
    main_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = main_split["test"]
    remaining_dataset = main_split["train"]

    # Now split the remaining into Train (80% total) and Val (10% total)
    train_val_split = remaining_dataset.train_test_split(test_size=0.1/0.9, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset   = train_val_split["test"]

    # Note: Class balancing is handled via WeightedRandomSampler in trainer.py
    # This avoids duplicating data in memory

    # 5. Apply the formatting function with valid classes
    formatting_func = make_formatting_func(valid_classes)
    train_dataset = train_dataset.map(formatting_func, batched=True)
    val_dataset   = val_dataset.map(formatting_func, batched=True)
    test_dataset  = test_dataset.map(formatting_func, batched=True)

    return train_dataset, val_dataset, test_dataset
