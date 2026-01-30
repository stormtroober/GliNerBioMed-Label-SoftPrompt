# -*- coding: utf-8 -*-
"""
Generazione dataset token-level BIO-aware con allineamento subtoken
per GLiNER-BioMed (bi-encoder token-level training).
Include generazione di label2desc.json e label2id.json.
Supporta generazione Multipla (Mono e Bi-Encoder).

Dataset: AnatEM (Anatomy Named Entity Recognition)
"""

import polars as pl
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
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_anatEM.conf')
config.read(config_path)

TRAIN_DATASET_SIZE = int(config['dataset']['TRAIN_DATASET_SIZE'])
VALIDATION_DATASET_SIZE = int(config['dataset']['VALIDATION_DATASET_SIZE'])
TEST_DATASET_SIZE = int(config['dataset']['TEST_DATASET_SIZE'])

# ===============================================================
# 1️⃣ GENERAZIONE label2desc.json e label2id.json per AnatEM
# ===============================================================
print("📥 Caricamento Pile-NER descrizioni...")
df_labels = pd.read_parquet("hf://datasets/disi-unibo-nlp/Pile-NER-biomed-descriptions/data/train-00000-of-00001.parquet")
print(f"✅ Pile-NER descrizioni caricato — {len(df_labels)} descrizioni totali")

# Etichette di interesse per AnatEM
target_labels = {"anatomy"}

# Filtra solo le descrizioni corrispondenti
df_labels_filtered = df_labels[df_labels["entity_type"].str.lower().isin(target_labels)]
print(f"✅ Descrizioni filtrate: {len(df_labels_filtered)} (Target: {target_labels})")

# Crea dizionari label2desc e label2id
label2desc = {
    row["entity_type"].lower(): row["description"]
    for _, row in df_labels_filtered.iterrows()
}
# Aggiungi descrizione per l'etichetta "O" (outside/non-entity)
label2desc["O"] = "Tokens that do not belong to any named entity. These are regular words, punctuation, or other text elements that are not part of any anatomical entity."

# Crea label2id 
label2id = {}
for i, lab in enumerate(label2desc.keys()):
    if lab != "O":
        label2id[lab] = i
# Aggiungi "O" con ID successivo
label2id["O"] = len(label2id)

# Salva su disco
os.makedirs("dataset_anatEM", exist_ok=True)
with open("dataset_anatEM/label2desc.json", "w") as f:
    json.dump(label2desc, f, indent=2)
with open("dataset_anatEM/label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)

print("✅ dataset_anatEM/label2desc.json e dataset_anatEM/label2id.json salvati")
print(f"📊 Etichette definite: {list(label2id.keys())}")

# ===============================================================
# UTILS DI TOKENIZZAZIONE
# ===============================================================
BIO2BASE = {
    "ANATOMY": "anatomy",
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

def get_spans_from_bio(words, bio_tags, label2id):
    """
    Converte BIO tags e tokens in spans [start, end, label_id_str].
    Replica la logica di grouping consecutivi (come convert_dataset)
    ma rispetta i prefix B- per separare entità adiacenti dello stesso tipo.
    """
    spans = []
    if not words:
        return spans

    start_idx = 0
    
    # Parse first
    first_tag = bio_tags[0]
    p_pref, p_base = parse_bio_tag(first_tag)
    p_id = label2id.get(p_base, label2id.get("O", 1)) # Fallback a O=1 se manca
    p_id_str = str(p_id) 
    
    current_label = p_id_str
    current_base = p_base

    for i in range(1, len(words)):
        tag = bio_tags[i]
        c_pref, c_base = parse_bio_tag(tag)
        c_id = label2id.get(c_base, label2id.get("O", 1))
        c_id_str = str(c_id)

        # Decision limit:
        # 1. Label ID different -> SPLIT
        # 2. Label ID same, BUT current is B- (and not O) -> SPLIT (Separates touching entities)
        # Note: 'O' usually doesn't have B/I, just 'O'. So O spans merge.
        
        is_split = False
        if c_id_str != current_label:
            is_split = True
        elif c_pref == "B" and c_base != "O":
            is_split = True
            
        if is_split:
            # Close previous
            spans.append([start_idx, i-1, current_label])
            # Start new
            start_idx = i
            current_label = c_id_str
            current_base = c_base
        else:
            # Continue
            pass

    # Flush last
    spans.append([start_idx, len(words)-1, current_label])
    
    return spans

# ===============================================================
# 2️⃣ FUNZIONE CORE DI GENERAZIONE (con supporto validation)
# ===============================================================
def generate_for_model(model_name, output_suffix, label2id, df_train, df_val, df_test, 
                       train_size, val_size, test_size):
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

    out_train = f"dataset_anatEM/anatEM_train_tknlvl_{output_suffix}.json"
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    print(f"✅ Train set (token-level) salvato: {out_train} ({len(train_records)} samples)")

    # --- TRAIN SPAN SET ---
    print(f"⚙️  [TRAIN] Generazione dataset span-based...")
    full_span_train = []
    for _, row in df_train.iterrows():
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        spans = get_spans_from_bio(words, bio_tags, label2id)
        full_span_train.append({
            "tokenized_text": words,
            "ner": spans
        })
        
    final_span_train = full_span_train[:train_size]
    
    out_span_train = f"dataset_anatEM/anatEM_train_span_{output_suffix}.json"
    with open(out_span_train, "w", encoding="utf-8") as f:
        json.dump(final_span_train, f, indent=2, ensure_ascii=False)
    print(f"✅ Train set (span-based) salvato: {out_span_train} ({len(final_span_train)} samples)")

    # --- VALIDATION SET ---
    print(f"\n⚙️  [VALIDATION] Tokenizzazione e allineamento...")
    val_encoded = []
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Validation encoding"):
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        ex = encode_and_align_labels(words, bio_tags, tokenizer, label2id)
        val_encoded.append(ex)

    # Sampling deterministico 
    rng_val = random.Random(42)
    
    if len(val_encoded) > val_size:
        val_sampled = rng_val.sample(val_encoded, val_size)
    else:
        val_sampled = val_encoded

    # Build span for validation set with same sampling
    full_span_val = []
    for _, row in df_val.iterrows():
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        spans = get_spans_from_bio(words, bio_tags, label2id)
        full_span_val.append({
            "tokenized_text": words,
            "ner": spans
        })
        
    # Ensure 1:1 match
    assert len(full_span_val) == len(val_encoded), "Mismatch in valid row count between token and span logic!"
    
    # Sync sampling between token-level and span-level
    combined_val = list(zip(val_encoded, full_span_val))
    rng_val_2 = random.Random(42)
    if len(combined_val) > val_size:
        sampled_combined_val = rng_val_2.sample(combined_val, val_size)
    else:
        sampled_combined_val = combined_val
        
    # Unzip
    final_val_encoded, final_span_val = zip(*sampled_combined_val) if sampled_combined_val else ([], [])
    
    val_records = []
    for ex in final_val_encoded:
        input_ids = ex["input_ids"].tolist()
        labels = ex["labels"].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        val_records.append({"tokens": tokens, "labels": labels})

    out_val = f"dataset_anatEM/anatEM_val_tknlvl_{output_suffix}.json"
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)
    print(f"✅ Validation set (token-level) salvato: {out_val} ({len(val_records)} samples)")
        
    out_span_val = f"dataset_anatEM/anatEM_val_span_{output_suffix}.json"
    with open(out_span_val, "w", encoding="utf-8") as f:
        json.dump(list(final_span_val), f, indent=2, ensure_ascii=False)
    print(f"✅ Validation set (span-based) salvato: {out_span_val} ({len(final_span_val)} samples)")

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
    rng = random.Random(42)
    
    if len(test_encoded) > test_size:
        test_sampled = rng.sample(test_encoded, test_size)
    else:
        test_sampled = test_encoded

    # Build span for test set
    full_span_test = []
    for _, row in df_test.iterrows():
        words = list(row["tokens"])
        bio_tags = list(row["ner_tags"])
        if not words or not bio_tags or len(words) != len(bio_tags):
            continue
        spans = get_spans_from_bio(words, bio_tags, label2id)
        full_span_test.append({
            "tokenized_text": words,
            "ner": spans
        })
        
    # Ensure 1:1 match with test_encoded
    assert len(full_span_test) == len(test_encoded), "Mismatch in valid row count between token and span logic!"
    
    # Sync sampling
    combined = list(zip(test_encoded, full_span_test))
    rng_test_2 = random.Random(42)
    if len(combined) > test_size:
        sampled_combined = rng_test_2.sample(combined, test_size)
    else:
        sampled_combined = combined
        
    # Unzip
    final_test_encoded, final_span_test = zip(*sampled_combined) if sampled_combined else ([], [])
    
    # Build test_records from the synced sample
    test_records = []
    for ex in final_test_encoded:
        input_ids = ex["input_ids"].tolist()
        labels = ex["labels"].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        test_records.append({"tokens": tokens, "labels": labels})

    out_test = f"dataset_anatEM/anatEM_test_tknlvl_{output_suffix}.json"
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2, ensure_ascii=False)
    print(f"✅ Test set (token-level) salvato: {out_test} ({len(test_records)} samples)")
        
    out_span_test = f"dataset_anatEM/anatEM_test_span_{output_suffix}.json"
    with open(out_span_test, "w", encoding="utf-8") as f:
        json.dump(list(final_span_test), f, indent=2, ensure_ascii=False)
    print(f"✅ Test set (span-based) salvato: {out_span_test} ({len(final_span_test)} samples)")
    
    verify_dataset_stats(train_records, f"TRAIN ({output_suffix})")
    verify_dataset_stats(val_records, f"VALIDATION ({output_suffix})")

# ===============================================================
# 3️⃣ CARICAMENTO DATI RAW (Una volta sola) con Polars
# ===============================================================
print("\n📥 Caricamento dataset RAW AnatEM...")
splits = {
    'train': 'data/train-00000-of-00001.parquet', 
    'validation': 'data/validation-00000-of-00001.parquet', 
    'test': 'data/test-00000-of-00001.parquet'
}

# Load with Polars, convert to Pandas for compatibility with existing logic
df_train_pl = pl.read_parquet('hf://datasets/disi-unibo-nlp/AnatEM/' + splits['train'])
df_val_pl = pl.read_parquet('hf://datasets/disi-unibo-nlp/AnatEM/' + splits['validation'])
df_test_pl = pl.read_parquet('hf://datasets/disi-unibo-nlp/AnatEM/' + splits['test'])

df_train = df_train_pl.to_pandas()
df_val = df_val_pl.to_pandas()
df_test = df_test_pl.to_pandas()

print(f"✅ Dati Raw caricati:")
print(f"   Train: {len(df_train)} samples")
print(f"   Validation: {len(df_val)} samples")
print(f"   Test: {len(df_test)} samples")

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
        df_val=df_val,
        df_test=df_test,
        train_size=TRAIN_DATASET_SIZE,
        val_size=VALIDATION_DATASET_SIZE,
        test_size=TEST_DATASET_SIZE
    )

print("\n" + "="*60)
print("✅ TUTTI I DATASET AnatEM GENERATI CON SUCCESSO")
