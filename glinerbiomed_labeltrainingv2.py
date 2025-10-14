# -*- coding: utf-8 -*-
"""
Automatic Description Learning (FINAL - token-level bi-encoder)
===============================================================
Flow:
 1️⃣ Tokenizzazione BIO-aware → gestione subtoken alignment
 2️⃣ Text encoder (frozen)
 3️⃣ Label encoder + projection (trainabili)
 4️⃣ Similarità token↔descrizione
 5️⃣ CrossEntropyLoss per token

Dataset: dataset_tokenlevel_balanced.json
Label set: derivato da label2desc.json / label2id.json
"""

import json, torch, torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from gliner import GLiNER
from tqdm import tqdm
from collections import Counter

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
TEMPERATURE = 1.0
GRAD_CLIP = 1.0
SCHEDULER_STEP = 2
SCHEDULER_GAMMA = 0.5
RANDOM_SEED = 42

DATASET_PATH = "dataset_tokenlevel_balanced.json"
LABEL2DESC_PATH = "label2desc.json"
LABEL2ID_PATH = "label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
TEST_EXAMPLE_IDX = 5

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 0️⃣ MODELLO + TOKENIZER
# ==========================================================
print("📦 Caricamento modello base GLiNER-BioMed...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model       # text encoder (frozen)
lbl_enc = core.token_rep_layer.labels_encoder.model   # label encoder (trainable)
proj    = core.token_rep_layer.labels_projection      # projection (trainable)

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Congeliamo il text encoder
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = True
for p in proj.parameters(): p.requires_grad = True

# ==========================================================
# 1️⃣ LABELS E DESCRIZIONI
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())

def compute_label_matrix(label2desc: dict) -> torch.Tensor:
    """Embedda le descrizioni con lbl_enc + proj (trainabili)."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)   # [num_labels, hidden_dim]

# ==========================================================
# 2️⃣ DATASET TOKEN-LEVEL
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, label2id):
        with open(path_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]

        input_ids = self.tok.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        y = []
        for lab in labels:
            if lab == -100: y.append(-100)
            else: y.append(lab)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(y),
        }

def collate_batch(batch, pad_id, ignore_index=-100):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), ignore_index, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

# ==========================================================
# 3️⃣ TRAINING SETUP
# ==========================================================
print("📚 Caricamento dataset...")
ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))

def compute_class_weights(data_path, label2id):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    counts = torch.zeros(len(label2id))
    total = 0
    
    for record in data:
        for label in record["labels"]:
            if label != -100:
                counts[label] += 1
                total += 1
    
    weights = total / (len(label2id) * counts.clamp(min=1))
    print(f"🔧 Class weights: {weights}")
    return weights

class_weights = compute_class_weights(DATASET_PATH, label2id).to(DEVICE)
ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

optimizer = optim.Adam(list(lbl_enc.parameters()) + list(proj.parameters()), 
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

txt_enc.eval().to(DEVICE)
lbl_enc.train().to(DEVICE)
proj.train().to(DEVICE)

# ==========================================================
# 4️⃣ TRAINING LOOP
# ==========================================================
print("\n🚀 Inizio training (token-level bi-encoder)...\n")

for epoch in range(1, EPOCHS + 1):
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    class_correct = torch.zeros(len(label_names))
    class_total = torch.zeros(len(label_names))
    
    for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        # Hidden del testo (frozen)
        with torch.no_grad():
            out_txt = txt_enc(**{k: batch[k] for k in ["input_ids","attention_mask"]})
            H = F.normalize(out_txt.last_hidden_state, dim=-1)  # [B, T, D]

        # Hidden delle descrizioni (trainabili)
        label_matrix = compute_label_matrix(label2desc).to(DEVICE)  # [num_labels, D]

        # Similarità token-label + Temperature scaling
        logits = torch.matmul(H, label_matrix.T) / TEMPERATURE  # [B, T, num_labels]

        # Loss
        loss = ce(logits.view(-1, len(label_names)), batch["labels"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lbl_enc.parameters()) + list(proj.parameters()), GRAD_CLIP)
        optimizer.step()

        # Accuracy per classe
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        acc = (preds[mask] == batch["labels"][mask]).float().sum().item()
        total_acc += acc; n_tokens += mask.sum().item(); total_loss += loss.item()
        
        # Statistiche per classe
        for i in range(len(label_names)):
            class_mask = (batch["labels"] == i) & mask
            if class_mask.sum() > 0:
                class_correct[i] += (preds[class_mask] == i).sum().item()
                class_total[i] += class_mask.sum().item()

    scheduler.step()
    
    print(f"Epoch {epoch}/{EPOCHS} | loss={total_loss/len(loader):.4f} | acc={(total_acc/n_tokens)*100:.1f}%")
    
    # Accuracy per classe
    print("📊 Accuracy per classe:")
    for i, label_name in enumerate(label_names):
        if class_total[i] > 0:
            acc_class = (class_correct[i] / class_total[i]) * 100
            print(f"  {label_name:12s}: {acc_class:5.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

print("\n✅ Fine training.\n")

# ==========================================================
# 🔍 ANALISI 1: Distribuzione delle etichette nel dataset
# ==========================================================
print("\n=== 📊 ANALISI DATASET ===")
with open(DATASET_PATH, "r") as f:
    data = json.load(f)

all_labels = []
for record in data:
    for label in record["labels"]:
        if label != -100:
            all_labels.append(label)

label_counts = Counter(all_labels)
print(f"Totale token: {len(all_labels)}")
print(f"Etichette uniche: {len(label_counts)}")
print("\nDistribuzione top 10:")
for label_id, count in label_counts.most_common(10):
    label_name = id2label.get(label_id, f"ID_{label_id}")
    percentage = (count / len(all_labels)) * 100
    print(f"  {label_name:15s}: {count:6d} ({percentage:5.1f}%)")

# ==========================================================
# 🔍 ANALISI 2: Rappresentazioni delle etichette
# ==========================================================
print("\n=== 🧠 ANALISI RAPPRESENTAZIONI ===")
with torch.no_grad():
    label_matrix = compute_label_matrix(label2desc).to(DEVICE)
    
    # Similarità tra etichette
    similarities = torch.matmul(label_matrix, label_matrix.T)
    
    print(f"Shape label matrix: {label_matrix.shape}")
    print(f"Media similarità tra etichette: {similarities.mean():.4f}")
    print(f"Std similarità: {similarities.std():.4f}")
    
    # Etichette più simili
    similarities.fill_diagonal_(-1)  # Ignora diagonale
    max_sim_idx = similarities.argmax()
    i, j = max_sim_idx // len(label_names), max_sim_idx % len(label_names)
    print(f"Etichette più simili: {label_names[i]} ↔ {label_names[j]} (sim: {similarities[i,j]:.4f})")

# ==========================================================
# 🔍 ANALISI 3: Predizioni su batch di training
# ==========================================================
print("\n=== 🎯 ANALISI PREDIZIONI ===")
with torch.no_grad():
    # Prendi un batch dal training
    sample_batch = next(iter(loader))
    sample_batch = {k: v.to(DEVICE) for k, v in sample_batch.items()}
    
    # Forward pass
    out_txt = txt_enc(**{k: sample_batch[k] for k in ["input_ids","attention_mask"]})
    H = F.normalize(out_txt.last_hidden_state, dim=-1)
    label_matrix = compute_label_matrix(label2desc).to(DEVICE)
    logits = torch.matmul(H, label_matrix.T)
    
    # Analisi logits
    print(f"Shape logits: {logits.shape}")
    print(f"Media logits: {logits.mean():.4f}")
    print(f"Std logits: {logits.std():.4f}")
    print(f"Min/Max logits: {logits.min():.4f} / {logits.max():.4f}")
    
    # Distribuzione predizioni
    preds = logits.argmax(-1)
    mask = sample_batch["labels"] != -100
    pred_counts = Counter(preds[mask].cpu().numpy())
    
    print("\nPredizioni nel batch:")
    for pred_id, count in pred_counts.most_common(5):
        label_name = id2label.get(pred_id, f"ID_{pred_id}")
        print(f"  {label_name:15s}: {count:4d}")
    
    print("\nGround truth nel batch:")
    gt_counts = Counter(sample_batch["labels"][mask].cpu().numpy())
    for gt_id, count in gt_counts.most_common(5):
        label_name = id2label.get(gt_id, f"ID_{gt_id}")
        print(f"  {label_name:15s}: {count:4d}")

# ==========================================================
# 5️⃣ TEST SU FRASE DEL TRAINING
# ==========================================================
example = ds.records[TEST_EXAMPLE_IDX]
tokens_test = [t for t in example["tokens"] if t not in ["[CLS]", "[SEP]"]]
sentence = " ".join([t.replace("▁", "") for t in tokens_test])

print(f"\n🔍 Test su frase id={TEST_EXAMPLE_IDX}:\n{sentence[:200]}...\n")

enc = txt_tok(sentence.split(), is_split_into_words=True,
              return_tensors="pt", truncation=True, padding=True).to(DEVICE)

with torch.no_grad():
    H = F.normalize(txt_enc(**enc).last_hidden_state, dim=-1)
    L = compute_label_matrix(label2desc).to(DEVICE)
    sims = torch.matmul(H, L.T).squeeze(0)
    preds = sims.argmax(-1)

tokens = txt_tok.convert_ids_to_tokens(enc["input_ids"][0])
print("=== 🔬 PREDIZIONI FINALI ===")
for tok, p in zip(tokens, preds):
    if tok in ["[CLS]", "[SEP]"]: continue
    print(f"{tok:15s} → {id2label[p.item()]}")
