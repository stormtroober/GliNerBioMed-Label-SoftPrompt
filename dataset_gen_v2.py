# -*- coding: utf-8 -*-
"""
Generazione dataset token-level BIO-aware con mascherizzazione dei subtoken
per GLiNER-BioMed (bi-encoder token-level training).

Pipeline:
1️⃣ Legge il dataset JNLPBA (train o test)
2️⃣ Mantiene i tag BIO (B- / I- / O)
3️⃣ Tokenizza e allinea le label a livello di subtoken
    - B-XXX → label valida solo sul primo subtoken
    - I-XXX → label propagata anche ai subtoken interni
    - O     → fuori da entità
4️⃣ Bilancia il dataset
5️⃣ Esporta in JSON leggibile
6️⃣ Verifica automatica finale
"""

import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer
from collections import Counter, defaultdict
import random

# ===============================================================
# 1️⃣ CONFIGURAZIONE
# ===============================================================
print("📥 Caricamento label2id.json e tokenizer...")
with open("label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

# Dataset remoto HuggingFace (JNLPBA)
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
}
df_data = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["train"])
df = df_data.head(15000).copy()  # subset di test rapido
print(f"✅ Dataset caricato: {len(df)} righe")

# ===============================================================
# 2️⃣ MAPPATURA BIO → LABEL BASE
# ===============================================================
BIO2BASE = {
    "DNA": "dna",
    "PROTEIN": "protein",
    "CELL_TYPE": "cell type",
    "CELL LINE": "cell line",
    "CELL_LINE": "cell line",
    "RNA": "rna",
}

def parse_bio_tag(tag: str):
    """Ritorna (prefisso, label_base) — gestisce anche formati irregolari."""
    if tag == "O":
        return ("O", "O")
    pref, _, typ = tag.partition("-")
    base = BIO2BASE.get(typ.upper().replace("-", " "), typ.lower())
    return (pref, base or "O")

# ===============================================================
# 3️⃣ TOKENIZZAZIONE + MASCHERIZZAZIONE (BIO-AWARE)
# ===============================================================
def encode_and_align_labels(words, bio_tags, tokenizer, label2id):
    """
    Tokenizza e allinea le label a livello di subtoken.
    BIO-aware:
      - B-XXX → label sul primo subtoken
      - I-XXX → label sulle parole successive dello stesso tipo
      - Subtoken interni (della stessa parola) → -100
    """
    bio_info = [parse_bio_tag(t) for t in bio_tags]
    prefixes, base_labels = zip(*bio_info)

    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )
    word_ids = enc.word_ids(0)
    labels_subtok = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels_subtok.append(-100)
            continue

        pref, lbl = prefixes[word_idx], base_labels[word_idx]

        # Se è il primo subtoken della parola
        if word_idx != previous_word_idx:
            if pref in ["B", "I"] and lbl != "O":
                labels_subtok.append(label2id.get(lbl, label2id["O"]))
            else:
                labels_subtok.append(label2id["O"])
        else:
            # ⚠️ Subtoken interno della stessa parola → sempre -100
            labels_subtok.append(-100)

        previous_word_idx = word_idx

    enc = {k: v.squeeze(0) for k, v in enc.items()}
    enc["labels"] = torch.tensor(labels_subtok)
    return enc


# ===============================================================
# 4️⃣ COSTRUZIONE DEL DATASET
# ===============================================================
print("\n⚙️  Costruzione dataset token-level BIO-aware...")

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
label_counts = Counter()
for ex in encoded_dataset:
    for l in ex["labels"].tolist():
        if l != -100:
            label_counts[l] += 1

non_o_labels = [lid for lid in label_counts if id2label[lid] != "O"]
min_count = 100
balanced_examples = defaultdict(list)

for ex in encoded_dataset:
    labels = ex["labels"].tolist()
    for lid in set([l for l in labels if l != -100 and id2label[l] != "O"]):
        balanced_examples[lid].append(ex)

balanced_dataset = []
for lid in non_o_labels:
    random.shuffle(balanced_examples[lid])
    balanced_dataset.extend(balanced_examples[lid][:min_count])

print(f"\n✅ Dataset bilanciato con {len(balanced_dataset)} frasi ({min_count} per classe)")

# ===============================================================
# 6️⃣ ESPORTAZIONE IN JSON
# ===============================================================
records = []
for i, ex in enumerate(balanced_dataset):
    tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"], skip_special_tokens=True)
    labels = [
        id2label.get(l, "IGNORE") if l != -100 else "IGNORE"
        for l in ex["labels"].tolist()
    ]
    records.append({"id": i, "tokens": tokens, "labels": labels})

with open("dataset_masked_balanced_bio.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"💾 Salvato in: dataset_masked_balanced_bio.json ({len(records)} esempi)")

# ===============================================================
# 7️⃣ VERIFICA FINALE COMPLETA SU TUTTO IL DATASET
# ===============================================================
def verify_final_dataset(records, tokenizer, max_inconsistency_ratio=0.02):
    """
    Verifica completa dell'intero dataset.
      - controlla coerenza subtoken/label
      - valuta distribuzione mascherizzazione
      - calcola tasso di incongruenze
      - stampa validazione finale automatica
    """
    print(f"\n🧪 Verifica finale BIO-aware su {len(records)} frasi (intero dataset)...")
    total_tokens = 0
    ignored_tokens = 0
    valid_label_tokens = 0
    inconsistencies = []

    for i, rec in enumerate(records):
        tokens = rec["tokens"]
        labels = rec["labels"]

        for tok, lab in zip(tokens, labels):
            total_tokens += 1
            if lab == "IGNORE":
                ignored_tokens += 1
            else:
                valid_label_tokens += 1

            # Subtoken non iniziale con label valida → potenziale errore
            if not tok.startswith("▁") and lab not in ["IGNORE", "O"]:
                # ignora numeri, acronimi, punteggiatura
                if (
                    any(c.isdigit() for c in tok)
                    or tok.isupper()
                    or len(tok) <= 2
                    or tok in ["-", "/", "(", ")", "’", "'", '"', ".", ",", ":"]
                ):
                    continue
                inconsistencies.append((i, tok, lab))

    # Statistiche globali
    mask_ratio = ignored_tokens / total_tokens if total_tokens else 0
    valid_ratio = valid_label_tokens / total_tokens if total_tokens else 0
    inconsistency_ratio = len(inconsistencies) / total_tokens if total_tokens else 0

    print("\n📊 STATISTICHE GLOBALI")
    print(f"   • Token totali analizzati: {total_tokens}")
    print(f"   • Token mascherati (-100 / IGNORE): {ignored_tokens} ({mask_ratio*100:.2f}%)")
    print(f"   • Token con label valida: {valid_label_tokens} ({valid_ratio*100:.2f}%)")
    print(f"   • Incongruenze trovate: {len(inconsistencies)} ({inconsistency_ratio*100:.3f}%)")

    # Risultato finale
    print("\n🧩 RISULTATO FINALE")
    if inconsistency_ratio <= max_inconsistency_ratio:
        print(f"✅ Dataset approvato: incongruenze entro la soglia ({max_inconsistency_ratio*100:.1f}%)")
    else:
        print(f"⚠️ Dataset da rivedere: troppe incongruenze ({inconsistency_ratio*100:.2f}%)")

    # Esempi campione di incongruenze (se presenti)
    if inconsistencies:
        print("\n⚠️  Prime 10 incongruenze rilevate:")
        for i, tok, lab in inconsistencies[:10]:
            print(f"  • Frase {i}: subtoken '{tok}' → '{lab}'")

# 🔧 Esegui la verifica completa
verify_final_dataset(records, tokenizer, max_inconsistency_ratio=0.02)

print("\n✅ Fine generazione e verifica completa dataset BIO-aware.")
