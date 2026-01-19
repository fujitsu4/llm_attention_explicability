"""
prepare_dbpedia.py
Author: Zakaria JOUILIL

Description:
    Prepare the DBpedia Ontology dataset by:
        - Cleaning text (whitespace + unicode normalization)
        - Removing sentences < 8 words
        - Removing sentences with semicolons (incompatible with SpaCy tokenization logic)
        - Keeping only sentences with exactly one syntactic ROOT
        - Removing sentences containing POS = 'X'
        - Removing sentences with incoherent punctuation (POS = 'PUNCT' but not a real punctuation)
        - Removing sentences with non-ASCII characters
Inputs:
    - None (dataset is downloaded automatically from HuggingFace)

Outputs:
    - data/dbpedia_filtered.csv
    - logs/rejected_dbpedia.txt   # rejected sentences

Usage:
    python -m src.prepare.prepare_dbpedia --output data/dbpedia_filtered.csv --target 10000
"""
from datasets import load_dataset
import pandas as pd
import spacy
import re
import argparse

# ----------------------------------------------------------
# CLI Parser
# ----------------------------------------------------------

parser = argparse.ArgumentParser(description="Prepare DBpedia dataset")
parser.add_argument("--output", type=str, default="data/dbpedia_filtered.csv",
                        help="Output CSV path")
parser.add_argument("--target", type=int, default=10000,
                        help="Number of sentences to keep")
args = parser.parse_args()

TARGET = args.target
OUTPUT = args.output

MIN_WORDS = 8
print("[INFO] Loading DBpedia Ontology dataset...")
ds = load_dataset("dbpedia_14", split="test")

nlp = spacy.load("en_core_web_sm")

sentences = []

# ----------------------------------------------------------
# 1) FUNCTION: clean a sentence BEFORE ANY processing
# ----------------------------------------------------------
def clean_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # remove tabs, non-breaking spaces, weird unicode spaces
    text = text.replace("\t", " ").replace("\xa0", " ")

    # collapse multiple spaces into a single one
    text = re.sub(r"\s+", " ", text)

    return text


# ----------------------------------------------------------
# 2) Collect all premises + hypotheses
# ----------------------------------------------------------

label_names = ds.features["label"].names
LABEL_CLASS_MAP = {i: name for i, name in enumerate(label_names)}

for row in ds:
    text = row["content"]
    label_id = row["label"]

    if text:
        text = clean_sentence(text)
        if len(text.split()) >= MIN_WORDS:
            sentences.append((text, label_id, LABEL_CLASS_MAP[label_id]))

print("[INFO] Collected raw sentences : ", len(sentences))

# ----------------------------------------------------------
# 3) Remove duplicates
# ----------------------------------------------------------
seen = {}
for text, label, label_class in sentences:
    if text not in seen:
        seen[text] = (label, label_class)

unique_sentences = [(text, label, label_class) for text, (label, label_class) in seen.items()]
len_unique = len(unique_sentences)
print("[INFO] After deduplication : ", len_unique)

# ----------------------------------------------------------
# 4) Helper functions
# ----------------------------------------------------------
def has_unique_root(sentence: str) -> bool:
    doc = nlp(sentence)
    roots = [tok for tok in doc if tok.dep_ == "ROOT"]
    return len(roots) == 1

def contains_semicolon(sentence: str) -> bool:
    return ";" in sentence

def contains_invalid_punct_token(sentence: str) -> bool:
    """
    Detect phrases containing tokens with POS = PUNCT but spacy thinks is_punct=False.
    This indicates an incoherent tokenization (e.g., '-An').
    """
    doc = nlp(sentence)
    for tok in doc:
        if tok.pos_ == "PUNCT" and tok.is_punct == False:
            return True
    return False

def contains_pos_x(sentence: str) -> bool:
    doc = nlp(sentence)
    return any(tok.pos_ == "X" for tok in doc)

def contains_non_ascii(sentence: str) -> bool:
    """
    Reject sentences containing any non-ASCII character.
    This removes soft hyphens, unicode dashes, exotic punctuation, etc.
    """
    return any(ord(c) > 127 for c in sentence)

# ----------------------------------------------------------
# 6) Filter sentences
# ----------------------------------------------------------
final_sentences = []
rejected_sentences = []

for s, label, label_class in unique_sentences:
    if contains_semicolon(s):
        rejected_sentences.append("[SEMICOLON] " + s)
        continue
    
    if contains_pos_x(s):
        rejected_sentences.append("[POS_X] " + s)
        continue
    
    if contains_invalid_punct_token(s):
        rejected_sentences.append("[INVALID_PUNCT] " + s)
        continue
    
    if contains_non_ascii(s):
        rejected_sentences.append("[ASCII] " + s)
        continue

    if has_unique_root(s):
        final_sentences.append((s, label, label_class))
        if len(final_sentences) >= TARGET:
            break
    else:
        rejected_sentences.append("[ROOT] " + s)

rejected_path = "logs/rejected_dbpedia.txt"
with open(rejected_path, "w", encoding="utf-8") as f:
    for r in rejected_sentences:
        f.write(f"{r}\n")

print(f"[INFO] Total sentences after ROOT filter : {len(final_sentences)}")
print(f"[INFO] Rejected sentences                : {len(rejected_sentences)}")
print(f"[INFO] Saving rejected sentences to      : {rejected_path}")

# ----------------------------------------------------------
# 7) Build dataframe
# ----------------------------------------------------------
df = pd.DataFrame([
    {
        "sentence_id": i,
        "sentence": s,
        "label": label,
        "label_class": label_class
    }
    for i, (s, label, label_class) in enumerate(final_sentences)
])

df.to_csv(OUTPUT, sep=";", index=False)
print(f"[INFO] Saving cleaned dataset to         : {OUTPUT}")