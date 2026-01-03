_MODEL_MAP = {
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "qwen3:8b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

def get_model_id(model_name: str) -> str:
    return _MODEL_MAP.get(model_name,model_name)
