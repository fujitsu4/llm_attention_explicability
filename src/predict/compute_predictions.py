"""
compute_predictions.py
Author: Zakaria JOUILIL

Description:
    Compute probabilistic prediction diagnostics for a fine-tuned BERT
    (on the cleaned Dbpedia dataset).

    For each sentence, the script computes the following LOCAL (instance-level)
    variables derived from the softmax probability distribution p(c):

    1) correct:
        correct = 1 if y_pred == y_true else 0

    2) p_true:
        p_true = p(y_true)
        → probability assigned by the model to the ground-truth class

    3) p_true_rank:
        p_true_rank = rank of p(y_true) among all class probabilities
        (1 = highest probability)

    4) confidence:
        confidence = max_c p(c)
        → model confidence in its predicted class

    5) margin (classification margin):
        margin = p(y_pred) - max_{c ≠ y_pred} p(c)
        → separation between the predicted class and the runner-up

    6) belief gap ):
        gap = p(y_pred) - p(y_true)
        → discrepancy between the predicted belief and the truth

    7) entropy:
        entropy = - Σ_c p(c) log p(c)
        → global uncertainty of the predictive distribution

Inputs:
    - data/dbpedia_filtered.csv

Outputs:
    - data/bert_predictions.csv

Usage:
    python -m src.predict.compute_predictions \
        --input data/dbpedia_filtered.csv \
        --output predict/dbpedia_predictions.csv
"""

import argparse
import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

parser = argparse.ArgumentParser(description="Compute BERT probabilistic diagnostics on DBpedia")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

df = pd.read_csv(args.input, sep=";")
sentences = df["sentence"].tolist()
y_true = df["label"].tolist()

print(f"[INFO] Loaded {len(sentences)} sentences")


# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------

MODEL_NAME = "textattack/bert-base-uncased-dbpedia"

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
print("DEBUG labels")
print(model.config.id2label)

# ----------------------------------------------------------
# Inference
# ----------------------------------------------------------

all_probs = []

with torch.no_grad():
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="Inference"):
        batch = sentences[i:i + args.batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits
        probs = softmax(logits, dim=-1)

        all_probs.extend(probs.cpu().numpy())


probs = np.array(all_probs)                     # [N, C]
y_pred = probs.argmax(axis=1)
confidence = probs.max(axis=1)
correct = (y_pred == np.array(y_true)).astype(int)

# ----------------------------------------------------------
# Compute probabilistic diagnostics
# ----------------------------------------------------------

p_true = probs[np.arange(len(probs)), y_true]

# rank of the true class probability
p_true_rank = np.argsort(-probs, axis=1).argsort(axis=1)[
    np.arange(len(probs)), y_true
] + 1

# margin = p_pred - second best
sorted_probs = np.sort(probs, axis=1)[:, ::-1]
margin = sorted_probs[:, 0] - sorted_probs[:, 1]

# belief gap
gap = confidence - p_true

# entropy
entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)


# ----------------------------------------------------------
# Build output dataframe
# ----------------------------------------------------------

df_out = pd.DataFrame({
    "sentence_id": df["sentence_id"],
    "sentence": df["sentence"],
    "y_true": y_true,
    "y_pred": y_pred,
    "correct": correct,
    "p_true": p_true,
    "p_true_rank": p_true_rank,
    "confidence": confidence,
    "margin": margin,
    "gap": gap,
    "entropy": entropy
})

df_out.to_csv(args.output, sep=";", index=False)

print(f"[INFO] Accuracy = {df_out['correct'].mean():.4f}")
print(f"[INFO] Saved predictions to {args.output}")
