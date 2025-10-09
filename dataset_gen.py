import pandas as pd
from tqdm import tqdm
import json

# ===============================================================
# 1️⃣ Caricamento dei dataset
# ===============================================================
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df_data = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["train"])

df_data_test = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["test"])

df_labels = pd.read_parquet("hf://datasets/disi-unibo-nlp/Pile-NER-biomed-descriptions/data/train-00000-of-00001.parquet")

print(f"✅ JNLPBA caricato — {len(df_data)} frasi totali")
print(f"✅ JNLPBA test caricato — {len(df_data_test)} frasi totali")
print(f"✅ Pile-NER descrizioni caricato — {len(df_labels)} descrizioni totali")

# ===============================================================
# 2️⃣ Etichette di interesse
# ===============================================================
bio2label = {
    "CELL_LINE": "cell line",
    "CELL_TYPE": "cell type",
    "DNA": "dna",
    "PROTEIN": "protein",
    "RNA": "rna"
}

target_labels = set(bio2label.values())

# Filtra solo le descrizioni corrispondenti
df_labels_filtered = df_labels[df_labels["entity_type"].str.lower().isin(target_labels)]
print(f"✅ Descrizioni filtrate: {len(df_labels_filtered)}")

# Crea dizionari label2desc e label2id
label2desc = {
    row["entity_type"].lower(): row["description"]
    for _, row in df_labels_filtered.iterrows()
}
label2id = {lab: i for i, lab in enumerate(label2desc.keys())}

# Salva su disco
with open("label2desc.json", "w") as f: json.dump(label2desc, f, indent=2)
with open("label2id.json", "w") as f: json.dump(label2id, f, indent=2)

print("✅ label2desc.json e label2id.json salvati")

# ===============================================================
# 3️⃣ Conversione BIO → (text, entity_span, label_id)
# ===============================================================
rows = []
for _, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Convertendo BIO → span"):
    tokens = list(row["tokens"])
    tags = list(row["ner_tags"])
    text = " ".join(tokens)

    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            label_raw = tag[2:].upper()  # ✅ fix qui
            label = bio2label.get(label_raw, None)
            if label is None or label not in label2id:
                i += 1
                continue
            start = i
            j = i + 1
            while j < len(tags) and tags[j].startswith("I-"):
                j += 1
            end = j - 1
            label_id = label2id[label]
            rows.append((text, (start, end), label_id))
            i = j
        else:
            i += 1

df_train = pd.DataFrame(rows, columns=["text", "entity_span", "label_id"])
#df_train.to_csv("train_data_subword.csv", index=False)
print(f"✅ Salvato train_data_subword.csv con {len(df_train)} istanze totali")

counts = df_train["label_id"].value_counts().sort_index()
print("\n=== 📊 Distribuzione classi ===")
for lab, lid in label2id.items():
    print(f"{lab:<12} → {counts.get(lid,0)} esempi")


# ===============================================================
# 5️⃣ (Bilanciamento ridotto per training rapido)
# ===============================================================
target_per_class = 500  # ✅ 200 esempi per classe

balanced = (
    df_train.groupby("label_id")
    .apply(lambda x: x.sample(n=min(len(x), target_per_class), random_state=42))
    .reset_index(drop=True)
)
balanced.to_csv("train_data_balanced.csv", index=False)

print(f"✅ train_data_balanced.csv salvato ({len(balanced)} esempi, max {target_per_class} per classe)")

counts = balanced["label_id"].value_counts().sort_index()
print("\n=== 📊 Distribuzione classi ===")
for lab, lid in label2id.items():
    print(f"{lab:<12} → {counts.get(lid,0)} esempi")


df_data_test = pd.read_parquet("hf://datasets/disi-unibo-nlp/JNLPBA/" + splits["test"])

print(f"✅ JNLPBA test caricato — {len(df_data_test)} frasi totali")


rows_test = []
for _, row in tqdm(df_data_test.iterrows(), total=len(df_data_test), desc="Convertendo TEST BIO → span"):
    tokens = list(row["tokens"])
    tags = list(row["ner_tags"])
    text = " ".join(tokens)

    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            label_raw = tag[2:].upper()
            label = bio2label.get(label_raw, None)
            if label is None or label not in label2id:
                i += 1
                continue
            start = i
            j = i + 1
            while j < len(tags) and tags[j].startswith("I-"):
                j += 1
            end = j - 1
            label_id = label2id[label]
            rows_test.append((text, (start, end), label_id))
            i = j
        else:
            i += 1

df_test = pd.DataFrame(rows_test, columns=["text", "entity_span", "label_id"])

# --- Distribuzione classi ---
counts_test = df_test["label_id"].value_counts().sort_index()
print("\n=== 📊 Distribuzione classi (TEST) ===")
for lab, lid in label2id.items():
    print(f"{lab:<12} → {counts_test.get(lid,0)} esempi")

# --- Selezione casuale di 300 esempi ---
test_sample_size = 500
df_test_sampled = df_test.sample(n=min(len(df_test), test_sample_size), random_state=42)
df_test_sampled.to_csv("test_data_random.csv", index=False)

print(f"✅ test_data_random.csv salvato ({len(df_test_sampled)} esempi totali)")

counts_sampled = df_test_sampled["label_id"].value_counts().sort_index()
print("\n=== 📊 Distribuzione classi (TEST random) ===")
for lab, lid in label2id.items():
    print(f"{lab:<12} → {counts_sampled.get(lid,0)} esempi")
