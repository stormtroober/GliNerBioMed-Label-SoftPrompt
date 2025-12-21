# -*- coding: utf-8 -*-
"""
Generazione dataset token-level BIO-aware con allineamento subtoken
per GLiNER-BioMed (bi-encoder token-level training).
Include generazione di label2desc.json e label2id.json.
Supporta generazione Multipla (Mono e Bi-Encoder).
"""

import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer
from collections import Counter
import random
import os
import configparser

# Configurazione dataset 
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.conf')
config.read(config_path)

TRAIN_DATASET_SIZE = int(config['dataset']['TRAIN_DATASET_SIZE'])
TEST_DATASET_SIZE = int(config['dataset']['TEST_DATASET_SIZE'])

# ===============================================================
# 1️⃣ GENERAZIONE label2desc.json e label2id.json
# ===============================================================
print("📥 Caricamento Pile-NER descrizioni...")
df_labels = pd.read_parquet("hf://datasets/disi-unibo-nlp/Pile-NER-biomed-descriptions/data/train-00000-of-00001.parquet")
print(f"✅ Pile-NER descrizioni caricato — {len(df_labels)} descrizioni totali")

# Etichette di interesse
target_labels = {"cell line", "cell type", "dna", "protein", "rna"}

# Filtra solo le descrizioni corrispondenti
df_labels_filtered = df_labels[df_labels["entity_type"].str.lower().isin(target_labels)]
print(f"✅ Descrizioni filtrate: {len(df_labels_filtered)} (Target: {target_labels})")

# Crea dizionari label2desc e label2id
label2desc = {
    row["entity_type"].lower(): row["description"]
    for _, row in df_labels_filtered.iterrows()
}
# Aggiungi descrizione per l'etichetta "O" (outside/non-entity)
label2desc["O"] = "Tokens that do not belong to any named entity. These are regular words, punctuation, or other text elements that are not part of any biological entity such as proteins, DNA, RNA, cell types, or cell lines."

# Crea label2id assicurandosi che "O" abbia ID 5
label2id = {}
for i, lab in enumerate(label2desc.keys()):
    if lab != "O":
        label2id[lab] = i
# Aggiungi "O" con ID 5
label2id["O"] = 5

# Salva su disco
with open("label2desc.json", "w") as f:
    json.dump(label2desc, f, indent=2)
with open("label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)

print("✅ label2desc.json e label2id.json salvati")
print(f"📊 Etichette definite: {list(label2id.keys())}")

# ===============================================================
# UTILS DI TOKENIZZAZIONE
# ===============================================================
BIO2BASE = {
    "DNA": "dna",
    "PROTEIN": "protein",
    "CELL_TYPE": "cell type",
    "CELL_LINE": "cell line",
    "RNA": "rna",
}

def parse_bio_tag(tag: str):
    if tag == "O": return ("O", "O")
    pref, _, typ = tag.partition("-")
    base = BIO2BASE.get(typ.upper().replace("-", " "), typ.lower())
    return (pref, base or "O")

def encode_and_align_labels(words, bio_tags, tokenizer, label2id):
    """
    Tokenizza e allinea le label a livello di subtoken.
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

def verify_dataset_stats(records, name):
    total, ignored, valid = 0, 0, 0
    for rec in records:
        for lab in rec["labels"]:
            total += 1
            if lab == -100: ignored += 1
            else: valid += 1
    if total > 0:
        print(f"📊 {name} Stats - Valid: {valid/total*100:.1f}% | Ignored: {ignored/total*100:.1f}%")

# ===============================================================
# 2️⃣ FUNZIONE CORE DI GENERAZIONE
# ===============================================================
def generate_for_model(model_name, output_suffix, label2id, df_train, df_test, train_size, test_size):
    print(f"\n" + "="*60)
    print(f"🤖 GENERAZIONE DATASET PER: {model_name}")
    print(f"📂 Suffix output: {output_suffix}")
    print("="*60)

    try:
        # Load specific tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"⚠️ Errore caricamento tokenizer per {model_name}: {e}")
        return

    # --- TRAIN SET ---
    print(f"\n⚙️  [TRAIN] Tokenizzazione e allineamento...")
    encoded_dataset = []
    
    # Process only enough records to satisfy TRAIN_DATASET_SIZE * buffer? 
    # To correspond exactly to previous logic, we process ALL df (expensive?) 
    # or just enough. Previous script processed ALL. Let's process ALL.
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Train encoding"):
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        ex = encode_and_align_labels(words, bio_tags, tokenizer, label2id)
        encoded_dataset.append(ex)

    final_dataset = encoded_dataset[:train_size]
    
    train_records = []
    for ex in final_dataset:
        input_ids = ex["input_ids"].tolist()
        labels = ex["labels"].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        train_records.append({"tokens": tokens, "labels": labels})

    out_train = f"dataset/dataset_tknlvl_{output_suffix}.json"
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    print(f"✅ Train set salvato: {out_train} ({len(train_records)} samples)")

    # --- TEST SET ---
    print(f"\n⚙️  [TEST] Tokenizzazione e allineamento...")
    test_encoded = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test encoding"):
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        ex = encode_and_align_labels(words, bio_tags, tokenizer, label2id)
        test_encoded.append(ex)

    # Sampling deterministico per confronto equo
    # Usiamo un generatore locale per non influenzare lo stato globale
    rng = random.Random(42)
    
    if len(test_encoded) > test_size:
        test_sampled = rng.sample(test_encoded, test_size)
    else:
        test_sampled = test_encoded

    test_records = []
    for ex in test_sampled:
        input_ids = ex["input_ids"].tolist()
        labels = ex["labels"].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        test_records.append({"tokens": tokens, "labels": labels})

    out_test = f"dataset/test_dataset_tknlvl_{output_suffix}.json"
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2, ensure_ascii=False)
    print(f"✅ Test set salvato: {out_test} ({len(test_records)} samples)")
    
    verify_dataset_stats(train_records, f"TRAIN ({output_suffix})")

# ===============================================================
# 3️⃣ CARICAMENTO DATI RAW (Una volta sola)
# ===============================================================
print("\n📥 Caricamento dataset RAW JNLPBA...")
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
}
# Train Data
df_train = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["train"])

# Test Data (Load enough to sample from)
df_test_raw = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["test"])
df_test = df_test_raw.head(5000).copy() 

print(f"✅ Dati Raw caricati. Train: {len(df_train)} | Test Pool: {len(df_test)}")

# ===============================================================
# 4️⃣ ESECUZIONE GENERAZIONE MULTIPLA
# ===============================================================

# CONFIGURAZIONI TARGET
configs = [
    {
        "model": "microsoft/deberta-v3-small", 
        "suffix": "mono",
        "description": "Per Mono-Encoder (urchade/gliner_small-v2.1)"
    },
    {
        "model": "Ihor/gliner-biomed-bi-small-v1.0", 
        "suffix": "bi",
        "description": "Per Bi-Encoder (Ihor/gliner-biomed-bi-small-v1.0)"
    }
]

# Suppress tokenizer warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.convert_slow_tokenizer")

for cfg in configs:
    generate_for_model(
        model_name=cfg["model"],
        output_suffix=cfg["suffix"],
        label2id=label2id,
        df_train=df_train,
        df_test=df_test,
        train_size=TRAIN_DATASET_SIZE,
        test_size=TEST_DATASET_SIZE
    )

print("\n" + "="*60)
print("✅ TUTTI I DATASET GENERATI CON SUCCESSO")
