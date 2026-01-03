from datasets import load_dataset
from .config import TrainingConfig

# This data loader works only for the math dataset for now

def formatting_prompts_func(examples):
    """
    Formats the raw dataset into an instruction-tuning format.
    Unsloth/TRL will automatically train on the 'text' field.
    """
    instruction = "Classify the following math problem into its category (e.g., Algebra, Geometry, Number Theory)."

    problems = examples["problem"]
    types    = examples["type"]
    texts    = []
    prompts  = []

    for problem, type_ in zip(problems, types):
        # The standard Alpaca/Instruction format
        prompt = f"Instruction: {instruction}\nInput: {problem}\nOutput: "
        text   = f"{prompt}{type_}"

        texts.append(text)
        prompts.append(prompt)

    return { "text" : texts, "prompt" : prompts }

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

    # 3. Split into Train (90%) and Test (10%)
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset_dict["train"]
    test_dataset  = dataset_dict["test"]

    # 4. Apply the formatting function
    # batched=True is much faster as it uses multithreading internally
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    test_dataset  = test_dataset.map(formatting_prompts_func, batched=True)

    return train_dataset, test_dataset
