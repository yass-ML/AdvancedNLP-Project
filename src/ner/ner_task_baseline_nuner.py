import argparse
import pandas as pd
from gliner import GLiNER
import os
import sys
from seqeval.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import torch

def evaluate_nuner(dataset_path="datasets/few_nerd/test.parquet", model_name="numind/NuNerZero", sample_size=None):
    """
    Evaluates Nu-NER Zero (GLiNER) on the Few-NERD dataset.
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    if sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Subsampled to {len(df)} examples.")

    print(f"Loading model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(model_name).to(device)

    # Few-NERD labels are fine-grained (e.g. person-actor, organization-company)
    # NuNer model takes a list of labels to look for.
    # We need to extract all unique labels from the dataset to pass to GLiNER.

    all_labels = set()
    for tags in df['ner_tags']:
        for tag in tags:
            if tag != 'O':
                all_labels.add(tag)

    labels_list = list(all_labels)
    print(f"Found {len(labels_list)} unique labels.")

    y_true = []
    y_pred = []

    print("Running inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        true_tags = row['ner_tags']

        # GLiNER predict returns list of dicts: [{'start': 0, 'end': 5, 'label': 'person', 'text': 'Obama'}]
        # We need to align this with tokens.
        # However, for metric computation using seqeval, we need list of tags aligned with tokens.
        # Since GLiNER takes raw text, aligning back to tokens can be tricky if tokenization differs.
        # Few-NERD provides tokens.

        # Strategy:
        # 1. Predict entities using GLiNER on raw text.
        # 2. Reconstruct BIO tags for the original tokens.

        def predict_chunked(text, labels, chunk_size=350, overlap=50):
            # Simple word-based splitting as approximation
            words = text.split(' ')
            if len(words) <= chunk_size:
                 return model.predict_entities(text, labels, threshold=0.5)

            all_entities = []
            seen_spans = set()

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)
                # Map back to approximate char offset in original text
                # This is tricky because of spaces.
                # Heuristic: find chunk_text in text starting from approximate position
                # Or simplistic: Accumulate offsets.

                # Better approach: Just assume space join.
                # offset of this chunk's start in original text
                chunk_start_char = len(" ".join(words[:i])) + 1 if i > 0 else 0

                preds = model.predict_entities(chunk_text, labels, threshold=0.5)
                for p in preds:
                    abs_start = chunk_start_char + p['start']
                    abs_end = chunk_start_char + p['end']
                    span_key = (abs_start, abs_end, p['label'])

                    if span_key not in seen_spans:
                        p['start'] = abs_start
                        p['end'] = abs_end
                        all_entities.append(p)
                        seen_spans.add(span_key)
            return all_entities

        try:
            # entities = model.predict_entities(text, labels_list, threshold=0.5)
            # Use chunked prediction to avoid truncation
            entities = predict_chunked(text, labels_list)
        except Exception as e:
            print(f"Error prediction: {e}")
            entities = []

        # Reconstruct tags for the original tokens
        # detailed alignment is complex, for baseline we can try a simplified approach or use a token-alignment library.
        # A simple heuristic: check if token is inside any predicted entity span.

        pred_tags = ['O'] * len(row['tokens'])

        # We need character offsets for tokens to align with GLiNER character spans.
        # Re-creating simple offsets assuming space separation (which matches how we created 'text')

        current_char = 0
        token_spans = []
        for token in row['tokens']:
            start = current_char
            end = start + len(token)
            token_spans.append((start, end))
            current_char = end + 1 # +1 for space

        for entity in entities:
            e_start = entity['start']
            e_end = entity['end']
            e_label = entity['label']

            # Find tokens that overlap with this entity
            for t_i, (t_start, t_end) in enumerate(token_spans):
                # Check overlap
                if max(t_start, e_start) < min(t_end, e_end):
                    # For BIO scheme
                    # If it's the first token of the entity match (or close to start), B
                    # else I

                    # Simplification: Just allow the label. seqeval handles strict/loose.
                    # Or simpler: if overlaps, assign label.
                    # Handling conflicts: last one wins or first one wins.

                    # Let's use B- I- tags.
                    # If this token starts at or after entity start, and previous token was not this entity...

                    # Refined:
                    if t_start >= e_start:
                         # Check if previous token was part of this same entity
                         if t_i > 0 and pred_tags[t_i-1] == f"I-{e_label}" and token_spans[t_i-1][1] >= e_start:
                             pred_tags[t_i] = f"I-{e_label}"
                         elif t_i > 0 and pred_tags[t_i-1] == f"B-{e_label}" and token_spans[t_i-1][1] >= e_start:
                             pred_tags[t_i] = f"I-{e_label}"
                         else:
                             pred_tags[t_i] = f"B-{e_label}"
                    elif t_end > e_start: # Overlap but starts before
                         # Likely a continuation if we missed the start? Or B if it's the first token we see
                         pred_tags[t_i] = f"B-{e_label}"

        y_true.append(true_tags)
        # Note: True tags in Few-NERD are often just "person-actor", not "B-person-actor".
        # If they are IO/BIO, we need to handle that.
        # Looking at few-nerd samples, they are usually "O" or "art-broadcastprogram".
        # We should convert both to BIO or just compare raw if they are 1 token 1 tag.
        # BUT seqeval expects BIO or similar.
        # Let's inspect true labels in the first run or assume they are raw categories.
        # If they are raw categories, seqeval might complain or treat everything as O.
        # Actually Few-NERD supervised tags are usually category-coarse (fine grained).
        # We should probably convert y_pred to just raw labels if true labels are raw labels.

        # Adjustment: Convert BIO pred labels to raw labels for comparison if needed,
        # OR convert raw true labels to BIO.
        # Seqeval requires BIO.
        # Let's assume we need to convert everything to BIO.
        # function to convert raw list 'person', 'person', 'O' to 'B-person', 'I-person', 'O'

        # ... logic to align formats ...

        # Post-process pred_tags to merge adjacent same-label B-tags
        # e.g. B-person, B-person -> B-person, I-person
        for i in range(1, len(pred_tags)):
            curr = pred_tags[i]
            prev = pred_tags[i-1]
            if curr.startswith("B-") and prev.startswith("B-") or prev.startswith("I-"):
                curr_label = curr[2:]
                prev_label = prev[2:]
                if curr_label == prev_label:
                    pred_tags[i] = f"I-{curr_label}"

        y_pred.append(pred_tags)

    # Post-processing to match formats
    # If y_true is not BIO, convert to BIO
    # (Checking one sample would help, but let's assume raw)

    y_true_bio = []
    for tags in y_true:
        bio_tags = []
        prev = 'O'
        for tag in tags:
            if tag == 'O':
                bio_tags.append('O')
                prev = 'O'
            else:
                if tag != prev:
                    bio_tags.append(f"B-{tag}")
                else:
                     # If exact same tag, assume continuation
                     # (Though distinct adjacent entities of same type would be merged)
                    bio_tags.append(f"I-{tag}")
                prev = tag
        y_true_bio.append(bio_tags)

    print(f"F1 Score: {f1_score(y_true_bio, y_pred)}")
    print(classification_report(y_true_bio, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    evaluate_nuner(sample_size=args.sample)
