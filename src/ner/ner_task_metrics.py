import pandas as pd
from seqeval.metrics import f1_score, classification_report
import requests
import argparse
import os
import json
from tqdm import tqdm
import re
import torch
import sys
import os
# Add parent directory to path to allow importing shot_selector from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Reuse the ShotSelector from the math task, but we might need to patch/extend it for NER data format
from shot_selector import ShotSelector
import yaml
import time

class NERMetricsPipeline:
    def __init__(self, model_name="llama3:8b", dataset_path="datasets/few_nerd/test.parquet",
                 selector_strategy="random", k_shots=0, dpo_model_path="dpo_selector_model_ner",
                 train_dataset_path="datasets/few_nerd/train.parquet", debug=False):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.selector_strategy = selector_strategy
        self.k_shots = k_shots
        self.dpo_model_path = dpo_model_path
        self.selector = None
        self.base_url = "http://localhost:11434/api/generate"
        self.train_dataset_path = train_dataset_path # For retrieval pool
        self.debug = debug

    def load_data(self):
        if self.debug: print(f"Loading test data from {self.dataset_path}...")
        self.df_test = pd.read_parquet(self.dataset_path)

        # Load train data for selector pool
        if self.k_shots > 0:
            if self.debug: print(f"Loading train data from {self.train_dataset_path} for selector...")
            self.df_train = pd.read_parquet(self.train_dataset_path)

            # Prepare data for Selector (needs 'problem' column usually)
            # ShotSelector expects 'problem'. We rename 'text' to 'problem' for compatibility.
            # And 'solution'/'type' -> 'entities' or 'tags'

            self.df_train_formatted = self.df_train.rename(columns={'text': 'problem'})
            # We might need to serialize tags to string for simple display?

            self.selector = ShotSelector(
                self.df_train_formatted,
                method=self.selector_strategy,
                k=self.k_shots,
                dpo_model_path=self.dpo_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    def format_examples(self, examples):
        formatted = "Here are some examples of NER extraction:\n\n"
        for i, ex in enumerate(examples):
             # ex has 'tokens', 'ner_tags', 'problem' (text)
             # formats tags
             tags = ex['ner_tags']
             tokens = ex['tokens']

             # Format as:
             # Text: <text>
             # Entities: <entity_list>

             # Extract entities for display
             # e.g. "Obama (person-actor)"
             # We can just list them.

             entities_formatted = []
             current_entity = []
             current_label = None

             # Simple tag parsing if they are raw labels or BIO
             # Few-NERD supervised is usually raw fine-grained tags per token ?
             # Let's check the data loader output... ah, it downloads 'supervised'.
             # In previous step, I saw tags like 'location', 'organization'.
             # We can just output the list of tags corresponding to words.

             # A better prompt format:
             # Word: Label
             # ...

             # OR JSON format which LLMs are good at.

             json_out = []
             for t, tag in zip(tokens, tags):
                 if tag != 'O':
                     json_out.append({"word": t, "label": tag})

             formatted += f"Example {i+1}:\nText: {ex['problem']}\nEntities: {json.dumps(json_out)}\n\n"
        return formatted

    def predict(self, text, examples=""):
        prompt = f"""
        Extract Named Entities from the following text.
        Focus on these types: person, organization, location, product, event, building, art, other.
        Return the output as a JSON list of objects, where each object has 'word' and 'label'.
        Only include entities that are NOT 'O'.

        {examples}

        Text: {text}
        JSON Output:
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                # "num_ctx": 4096,      # Ensure enough context window
                "num_predict": 2048    # Prevent infinite generation
            },
             "format": "json" # Force JSON mode if available or just hope
        }

        try:
            if self.debug: print(f"DEBUG: Sending request to Ollama... (Length: {len(prompt)})")
            # Set a long timeout (e.g., 5 minutes) to avoid silent hangs but allow catching network issues
            response = requests.post(self.base_url, json=payload, timeout=300)
            if self.debug: print("DEBUG: Response received from Ollama.")
            response.raise_for_status()
            data = response.json()
            return data.get("response", ""), data.get("eval_count", 0), data.get("prompt_eval_count", 0)
        except requests.exceptions.Timeout:
            print("DEBUG: Request timed out > 300s")
            return "Timeout", 0, 0
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "", 0, 0

    def evaluate(self, sample_size=None):
        if sample_size:
            data = self.df_test.sample(sample_size, random_state=42)
        else:
            data = self.df_test

        y_true = []
        y_pred = []

        total_latency = 0
        total_tokens = 0

        results_log = []

        print(f"Evaluating on {len(data)} samples...")
        for i, row in tqdm(data.iterrows(), total=len(data)):
            text = row['text']
            true_tags = row['ner_tags'] # List of tags

            # Select examples
            examples_str = ""
            if self.k_shots > 0 and self.selector:
                examples = self.selector.select(text, k=self.k_shots)
                examples_str = self.format_examples(examples)

            # Debug logging
            prompt_len = len(examples_str) + len(text) + 200 # approx
            if self.debug: print(f"Sample {i}: Prompt text length approx {prompt_len} chars")

            start = time.time()
            pred_str, n_tok, n_prompt = self.predict(text, examples_str)
            latency = time.time() - start

            total_latency += latency
            total_tokens += n_tok

            # Parse prediction
            # Expecting JSON list
            pred_tags = ['O'] * len(row['tokens'])

            try:
                # Basic JSON extraction
                # match json structure
                match = re.search(r'\[.*\]', pred_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    preds = json.loads(json_str)

                    # Align with tokens (heuristic)
                    # This is tricky without character offsets in LLM output.
                    # We assume LLM outputs words present in text.

                    for p in preds:
                        word = p.get('word', '')
                        label = p.get('label', '')
                        # find word in tokens
                        for t_idx, token in enumerate(row['tokens']):
                            if token == word: # Exact match
                                # Check if already assigned (simple collision handling)
                                if pred_tags[t_idx] == 'O':
                                    pred_tags[t_idx] = label # Use raw label or convert to BIO?
                                    # Let's stick to raw labels for now if true_tags are raw.
            except:
                pass

            y_true.append([t if t != 'O' else 'O' for t in true_tags]) # Clean up
            y_pred.append(pred_tags)

        # Convert to BIO for seqeval if needed
        # Seqeval really wants BIO-like tags or it might treat everything as O if it looks like just classes.
        # But 'person' is not standard BIO.
        # We will wrap them in 'I-' if they are not O.

        def to_bio(tags):
            return [f"I-{t}" if t != 'O' else 'O' for t in tags]

        y_true_bio = [to_bio(t) for t in y_true]
        y_pred_bio = [to_bio(t) for t in y_pred]

        f1 = f1_score(y_true_bio, y_pred_bio)
        print(f"F1 Score: {f1}")

        return {
            "f1": float(f1),
            "avg_latency": float(total_latency / len(data)),
            "model": self.model_name,
            "strategy": self.selector_strategy,
            "k": self.k_shots
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--strategy", default="random")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--train_dataset", default="datasets/few_nerd/train.parquet")
    args = parser.parse_args()

    pipeline = NERMetricsPipeline(model_name=args.model, k_shots=args.k, selector_strategy=args.strategy, train_dataset_path=args.train_dataset)
    pipeline.load_data()
    res = pipeline.evaluate(sample_size=args.sample)
    print(res)
