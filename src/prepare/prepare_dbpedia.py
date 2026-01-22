"""
prepare_dbpedia.py
Author: Zakaria JOUILIL

Select only sentences with multiple syntactic roots (spacy) (> 3 roots)
Usage:
    python -m src.prepare.prepare_dbpedia --output data/dbpedia_filtered.csv
"""
import argparse
import pandas as pd
import spacy
from tqdm import tqdm

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Filter DBpedia test sentences by syntactic complexity"
)
parser.add_argument("--output", type=str, required=True,
                    help="Output CSV file (filtered sentences)")
parser.add_argument("--min_roots", type=int, default=4,
                    help="Minimum number of ROOT nodes (default: 4)")
parser.add_argument("--min_length", type=int, default=20,
                    help="Minimum sentence length in words (default: 20)")
args = parser.parse_args()


# ----------------------------------------------------------
# Load spaCy model
# ----------------------------------------------------------

from datasets import load_dataset

print("[INFO] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------


print("[INFO] Loading DBpedia Ontology dataset...")
df = load_dataset("dbpedia_14", split="test")

print(f"[INFO] Loaded {len(df)} sentences")


# ----------------------------------------------------------
# Filtering
# ----------------------------------------------------------

label_names = df.features["label"].names

filtered_rows = []


for i, row in enumerate(tqdm(df, desc="Filtering")):
    assert "content" in row, "DBpedia schema mismatch"
    sentence = str(row["content"]).strip()
    label = row["label"]
    label_name = label_names[label]

    # Word count
    word_count = len(sentence.split())
    if word_count < args.min_length:
        continue

    doc = nlp(sentence)

    # Count ROOT dependencies
    root_count = sum(1 for token in doc if token.dep_ == "ROOT")

    if root_count >= args.min_roots:
        filtered_rows.append({
            "sentence_id": i,
            "sentence": sentence,
            "label": label,
            "label_name": label_name, 
            "root_count": root_count
        })


# ----------------------------------------------------------
# Save output
# ----------------------------------------------------------

df_out = pd.DataFrame(filtered_rows)
df_out.to_csv(args.output, sep=";", index=False)

print("[INFO] Filtering completed")
print(f"[INFO] Retained {len(df_out)} sentences")
print(f"[INFO] Saved to {args.output}")
