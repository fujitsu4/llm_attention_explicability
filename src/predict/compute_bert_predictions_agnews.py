"""
compute_bert_predictions_agnews.py
Author: Zakaria JOUILIL

Description:
    Compute final BERT predictions on the cleaned AG News dataset.
    For each sentence, the script extracts:
        - ground-truth label (y_true)
        - predicted label (y_pred)
        - prediction confidence (softmax max)
        - correctness (correct / incorrect)

Inputs:
    - data/agnews_filtered.csv

Outputs:
    - data/agnews_predictions.csv

Usage:
    python -m src.predict.compute_bert_predictions_agnews \
        --input data/agnews_filtered.csv \
        --output data/agnews_predictions.csv
"""

import argparse
import pandas as pd
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.nn.functional import softmax
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

parser = argparse.ArgumentParser(description="Compute BERT predictions on AG News")
parser.add_argument("--input", type=str, required=True,
                    help="Path to agnews_filtered.csv")
parser.add_argument("--output", type=str, required=True,
                    help="Output CSV with predictions")

args = parser.parse_args()


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

df = pd.read_csv(args.input, sep=";")

assert {"sentence_id", "sentence", "label"}.issubset(df.columns)

sentences = df["sentence"].tolist()
y_true = df["label"].tolist()

print(f"[INFO] Loaded {len(sentences)} sentences")


# ----------------------------------------------------------
# Load model & tokenizer
# ----------------------------------------------------------

print("[INFO] Loading tokenizer and model...")
MODEL_NAME = "textattack/bert-base-uncased-ag-news"

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

num_labels = model.config.num_labels
assert num_labels == len(set(y_true)), \
    "Mismatch between model num_labels and dataset labels"

print(f"[INFO] Model num_labels = {num_labels}")


# ----------------------------------------------------------
# Inference loop
# ----------------------------------------------------------

all_preds = []
all_confidences = []
all_logits = []
all_true_probs = []

batch_size = 16

with torch.no_grad():
    for i in tqdm(range(0, len(sentences), batch_size), desc="Inference"):
        batch_sentences = sentences[i:i + batch_size]

        enc = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc)
        logits = outputs.logits                       # [B, C]
        probs = softmax(logits, dim=-1)               # [B, C]

        conf, preds = torch.max(probs, dim=-1)
        y_true_batch = torch.tensor(y_true[i:i + len(batch_sentences)]).to(device)
        true_probs = probs[range(len(y_true_batch)), y_true_batch]

        all_true_probs.extend(true_probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_confidences.extend(conf.cpu().numpy().tolist())
        all_logits.extend(logits.cpu().numpy().tolist())


# ----------------------------------------------------------
# Build output dataframe
# ----------------------------------------------------------

df_out = pd.DataFrame({
    "sentence_id": df["sentence_id"],
    "sentence": df["sentence"],
    "y_true": y_true,
    "y_pred": all_preds,
    "p_true": all_true_probs,
    "confidence": all_confidences,
    "correct": [int(p == t) for p, t in zip(all_preds, y_true)]
})

# Optional: keep logits (useful later for margin / entropy)
for c in range(num_labels):
    df_out[f"logit_{c}"] = [log[c] for log in all_logits]


# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

df_out.to_csv(args.output, sep=";", index=False)

acc = df_out["correct"].mean()
print(f"[INFO] Accuracy = {acc:.4f}")
print(f"[INFO] Saved predictions to {args.output}")
