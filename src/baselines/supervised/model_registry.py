_MODEL_MAP = {
    # Llama 3
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "llama3:8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",

    # Mistral
    "mistral:7b": "mistralai/Mistral-7B-v0.3",
    "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",

    # Gemma
    "gemma:7b": "google/gemma-7b",
    "gemma:7b-instruct": "google/gemma-7b-it",

    # Phi-3
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",

    # Qwen 2 & 2.5
    "qwen2:7b": "Qwen/Qwen2-7B",
    "qwen2:7b-instruct": "Qwen/Qwen2-7B-Instruct",
    # "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    # "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",

}

def get_model_id(model_name: str) -> str:
    return _MODEL_MAP.get(model_name,model_name)
