# -*- coding: utf-8 -*-
"""
Generazione dataset token-level BIO-aware con allineamento subtoken
per GLiNER-BioMed (bi-encoder token-level training).

"""

import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer
from collections import Counter, defaultdict
import random

EXAMPLE_NUMBER_FOR_BALANCED = 50
# This means 200 * 5 classes = 1000 examples in the balanced dataset
TEST_SAMPLE_SIZE = 1000

# ===============================================================
# 1️⃣ CONFIGURAZIONE
# ===============================================================
print("📥 Caricamento label2id.json e tokenizer...")
with open("../label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

TOKENIZER_NAME = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
}
df_data = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["train"])
df = df_data.head(15000).copy()
print(f"✅ Dataset caricato: {len(df)} righe")

# ===============================================================
# 2️⃣ MAPPATURA BIO → LABEL BASE
# ===============================================================
BIO2BASE = {
    "DNA": "dna",
    "PROTEIN": "protein",
    "CELL_TYPE": "cell type",
    "CELL_LINE": "cell line",
    "RNA": "rna",
}

def parse_bio_tag(tag: str):
    if tag == "O":
        return ("O", "O")
    pref, _, typ = tag.partition("-")
    base = BIO2BASE.get(typ.upper().replace("-", " "), typ.lower())
    return (pref, base or "O")

# ===============================================================
# 3️⃣ TOKENIZZAZIONE + ALLINEAMENTO (BIO-AWARE)
# ===============================================================
def encode_and_align_labels(words, bio_tags, tokenizer, label2id):
    """
    Tokenizza e allinea le label a livello di subtoken.
      - B/I → label valida anche sui subtokens (no -100 interni)
      - Special tokens ([CLS]/[SEP]) → -100
    """
    assert len(words) == len(bio_tags)
    bio_info = [parse_bio_tag(t) for t in bio_tags]
    _, base_labels = zip(*bio_info)

    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
        add_special_tokens=True
    )
    word_ids = enc.word_ids(0)
    labels_subtok = []
    for word_idx in word_ids:
        if word_idx is None:
            labels_subtok.append(-100)
        else:
            lbl = base_labels[word_idx]
            labels_subtok.append(label2id.get(lbl, label2id["O"]))

    enc = {k: v.squeeze(0) for k, v in enc.items()}
    enc["labels"] = torch.tensor(labels_subtok, dtype=torch.long)
    return enc

# ===============================================================
# 4️⃣ COSTRUZIONE DEL DATASET
# ===============================================================
print("\nCostruzione dataset token-level allineato...")

encoded_dataset = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    words = list(row["tokens"])
    bio_tags = list(row["ner_tags"])
    if not words or not bio_tags or len(words) != len(bio_tags):
        continue
    ex = encode_and_align_labels(words, bio_tags, tokenizer, label2id)
    encoded_dataset.append(ex)

print(f"\n✅ Creati {len(encoded_dataset)} esempi token-level.")

# ===============================================================
# 5️⃣ BILANCIAMENTO CLASSI
# ===============================================================
# raggruppa frasi per classe se contengono almeno
# un'entità di quella classe. Per ogni classe si estraggono fino a
# EXAMPLE_NUMBER_FOR_BALANCED esempi casuali. Una stessa frase può
# comparire in più classi se contiene più entità diverse.

label_counts = Counter()
for ex in encoded_dataset:
    for l in ex["labels"].tolist():
        if l != -100:
            label_counts[l] += 1

non_o_labels = [lid for lid in label_counts if id2label[lid] != "O"]
if non_o_labels:
    min_count = min(label_counts[lid] for lid in non_o_labels)
    target_per_class = EXAMPLE_NUMBER_FOR_BALANCED

balanced_examples = defaultdict(list)
for ex in encoded_dataset:
    present = {l for l in ex["labels"].tolist() if l != -100 and id2label.get(l, "O") != "O"}
    for lid in present:
        balanced_examples[lid].append(ex)

balanced_dataset = []
for lid in non_o_labels:
    random.shuffle(balanced_examples[lid])
    balanced_dataset.extend(balanced_examples[lid][:target_per_class])

print(f"\n✅ Dataset bilanciato con {len(balanced_dataset)} frasi (~{target_per_class} per classe)")

# ===============================================================
# 6️⃣ ESPORTAZIONE IN JSON (SOLO tokens, labels)
# ===============================================================
records = []
for ex in balanced_dataset:
    input_ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    # Esportiamo SOLO i campi necessari al training:
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    records.append({
        "tokens": tokens,
        "labels": labels
    })

out_path = "dataset/dataset_tokenlevel_balanced.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"💾 Salvato in: {out_path} ({len(records)} esempi)")

# ===============================================================
# 7️⃣ VERIFICA
# ===============================================================
def verify_final_dataset(records):
    total, ignored, valid = 0, 0, 0
    for rec in records:
        for lab in rec["labels"]:
            total += 1
            if lab == -100:
                ignored += 1
            else:
                valid += 1
    print(f"\n📊 Tot token: {total} | Label valide: {valid/total*100:.1f}% | Ignorate: {ignored/total*100:.1f}%")

verify_final_dataset(records)
print("\n✅ Fine generazione dataset allineato.")

# ===============================================================
# 8️⃣ GENERAZIONE TEST SET (SOLO tokens, labels)
# ===============================================================
print("\n📥 Caricamento test split JNLPBA...")
df_test_raw = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["test"])
df_test = df_test_raw.head(3000).copy()
print(f"✅ Test set caricato: {len(df_test)} righe")

print("\n⚙️  Costruzione test set token-level allineato...")
test_encoded_dataset = []
for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test encoding"):
    words = list(row["tokens"])
    bio_tags = list(row["ner_tags"])
    if not words or not bio_tags or len(words) != len(bio_tags):
        continue
    ex = encode_and_align_labels(words, bio_tags, tokenizer, label2id)
    test_encoded_dataset.append(ex)

print(f"\n✅ Creati {len(test_encoded_dataset)} esempi test token-level.")

random.seed(42)
test_sampled = random.sample(test_encoded_dataset, TEST_SAMPLE_SIZE)

test_records = []
for ex in test_sampled:
    input_ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    test_records.append({
        "tokens": tokens,
        "labels": labels
    })

test_out_path = "dataset/test_dataset_tokenlevel.json"
with open(test_out_path, "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2, ensure_ascii=False)

print(f"💾 Test set salvato in: {test_out_path} ({len(test_records)} esempi)")

# Distribuzione test set (opzionale ma utile)
def verify_test_dataset(records):
    total, ignored, valid = 0, 0, 0
    label_dist = Counter()
    for rec in records:
        for lab in rec["labels"]:
            total += 1
            if lab == -100:
                ignored += 1
            else:
                valid += 1
                label_dist[lab] += 1
    print(f"\n📊 TEST SET - Tot token: {total} | Label valide: {valid/total*100:.1f}% | Ignorate: {ignored/total*100:.1f}%")
    print("\nDistribuzione etichette nel test set:")
    for label_id, count in label_dist.most_common():
        label_name = id2label.get(label_id, f"ID_{label_id}")
        percentage = (count / max(valid, 1)) * 100
        print(f"  {label_name:15s}: {count:6d} ({percentage:5.1f}%)")

verify_test_dataset(test_records)
print("\n✅ Fine generazione test set.")
