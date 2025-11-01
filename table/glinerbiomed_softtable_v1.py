# -*- coding: utf-8 -*-
"""
Training token-level con soft embeddings per GLiNER-BioMed.
Usa il dataset JSON generato con allineamento subtoken BIO-aware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from gliner import GLiNER
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random

# ==============================================================
# 1️⃣ CARICAMENTO MODELLO E CONFIGURAZIONE
# ==============================================================
print("📥 Caricamento modello GLiNER-BioMed...")
model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core = model.model

# Text encoder (DeBERTaV3)
txt_enc = core.token_rep_layer.bert_layer.model
# Label encoder (6-layer BERT, 384d)
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

# Congela text encoder
for p in txt_enc.parameters():
    p.requires_grad = False

print("✅ Modello caricato e text encoder congelato")

# ==============================================================
# 2️⃣ CARICAMENTO LABEL DEFINITIONS E CREAZIONE SOFT TABLE
# ==============================================================
print("\n📥 Caricamento label2id.json...")
with open("../label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# Leggi le descrizioni delle label dal file
with open("../label2desc.json") as f:
    label_descriptions = json.load(f)

# Ordina le descrizioni secondo label2id
labels = [label_descriptions[id2label[i]] for i in range(num_labels)]

print(f"✅ Caricate {num_labels} label")
for i, (name, desc) in enumerate(zip([id2label[j] for j in range(num_labels)], labels)):
    print(f"  [{i}] {name:15s}: {desc}")

# ==============================================================
# 3️⃣ TOKENIZZAZIONE DESCRIZIONI + HARD EMBEDDINGS (384d)
# ==============================================================
print("\n⚙️  Generazione hard embeddings dalle descrizioni...")
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)
batch = lbl_tok(labels, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    out = lbl_enc(**batch)
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
HE_384 = pooled  # [num_labels, 384]

print(f"✅ Hard embeddings generati: {HE_384.shape}")

# ==============================================================
# 4️⃣ CREAZIONE SOFT TABLE
# ==============================================================
print("\n⚙️  Creazione soft embedding table...")
soft_table = nn.Embedding(num_labels, HE_384.size(-1))
with torch.no_grad():
    soft_table.weight.copy_(HE_384)

# Registra nel modello e disattiva label encoder
core.token_rep_layer.soft_label_embeddings = soft_table
for p in core.token_rep_layer.labels_encoder.parameters():
    p.requires_grad = False

print("✅ Soft table creata e registrata nel modello")

# ==============================================================
# 5️⃣ DATASET CLASS
# ==============================================================
class TokenLevelDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        
        # Converti tokens in input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# ==============================================================
# 6️⃣ CARICAMENTO DATASET
# ==============================================================
print("\n📥 Caricamento dataset token-level...")
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)

train_dataset = TokenLevelDataset("dataset/dataset_tokenlevel_balanced.json", txt_tok)
test_dataset = TokenLevelDataset("dataset/test_dataset_tokenlevel.json", txt_tok)

print(f"✅ Train set: {len(train_dataset)} esempi")
print(f"✅ Test set: {len(test_dataset)} esempi")

# ==============================================================
# 7️⃣ DATALOADER CON COLLATE FUNCTION
# ==============================================================
def collate_fn(batch):
    """Padding dinamico per batch"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Padding
    max_len = max(len(ids) for ids in input_ids)
    
    padded_ids = []
    padded_labels = []
    attention_masks = []
    
    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), 
                                          torch.zeros(pad_len, dtype=torch.long)]))
    
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_labels)
    }

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==============================================================
# 8️⃣ TRAINING SETUP
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")

# Sposta modello su device
txt_enc.to(device)
soft_table.to(device)
proj.to(device)

# Loss e optimizer (solo soft_table + projection)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(
    list(soft_table.parameters()) + list(proj.parameters()),
    lr=1e-3,
    weight_decay=0.01
)

# ==============================================================
# 9️⃣ TRAINING LOOP
# ==============================================================
NUM_EPOCHS = 2

print("\n🚀 Inizio training...\n")
print("=" * 80)

for epoch in range(NUM_EPOCHS):
    # TRAINING
    soft_table.train()
    proj.train()
    txt_enc.eval()
    
    total_loss = 0
    correct = 0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 1. Text embeddings (frozen)
        with torch.no_grad():
            txt_out = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = txt_out.last_hidden_state  # [B, L, 768]
        
        # 2. Label embeddings (soft table + projection)
        label_vecs = proj(soft_table.weight)  # [num_labels, 768]
        
        # 3. Similarità normalizzata
        H_norm = F.normalize(H, dim=-1)
        label_norm = F.normalize(label_vecs, dim=-1)
        logits = torch.matmul(H_norm, label_norm.T)  # [B, L, num_labels]
        
        # 4. Loss
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        
        # 5. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metriche
        total_loss += loss.item()
        mask = labels_flat != -100
        if mask.sum() > 0:
            preds = logits_flat.argmax(-1)
            correct += ((preds == labels_flat) & mask).sum().item()
            total_tokens += mask.sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", 
                          "acc": f"{correct/max(total_tokens,1)*100:.1f}%"})
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct / max(total_tokens, 1) * 100
    
    # VALIDATION
    soft_table.eval()
    proj.eval()
    
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            txt_out = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = txt_out.last_hidden_state
            
            label_vecs = proj(soft_table.weight)
            
            H_norm = F.normalize(H, dim=-1)
            label_norm = F.normalize(label_vecs, dim=-1)
            logits = torch.matmul(H_norm, label_norm.T)
            
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            
            val_loss += loss.item()
            mask = labels_flat != -100
            if mask.sum() > 0:
                preds = logits_flat.argmax(-1)
                val_correct += ((preds == labels_flat) & mask).sum().item()
                val_total += mask.sum().item()
    
    val_loss /= len(test_loader)
    val_acc = val_correct / max(val_total, 1) * 100
    
    print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print("=" * 80)

# ==============================================================
# 🔟 SALVATAGGIO
# ==============================================================
print("\n💾 Salvataggio soft embeddings...")
torch.save({
    "soft_embeddings": soft_table.state_dict(),
    "projection": proj.state_dict(),
    "label2id": label2id,
    "id2label": id2label
}, "soft_embeddings_tokenlevel.pt")

print("✅ Training completato e modello salvato!")

# ==============================================================
# 1️⃣1️⃣ TEST INFERENCE
# ==============================================================
print("\n🧪 TEST INFERENCE")
print("-" * 80)

test_sentence = "Aspirin inhibits NF-kappa B activation in T cells."
enc = txt_tok(test_sentence, return_tensors="pt", padding=False, truncation=True)

with torch.no_grad():
    enc = {k: v.to(device) for k, v in enc.items()}
    out_txt = txt_enc(**enc)
    H = F.normalize(out_txt.last_hidden_state, dim=-1)
    
    label_vecs_768 = proj(soft_table.weight)
    label_matrix = F.normalize(label_vecs_768, dim=-1)
    
    logits = torch.matmul(H, label_matrix.T).squeeze(0)
    preds = logits.argmax(-1).cpu().tolist()

tokens = txt_tok.convert_ids_to_tokens(enc["input_ids"][0])
pred_labels = [id2label[p] for p in preds]

for tok, lab in zip(tokens, pred_labels):
    print(f"{tok:<25} →  {lab}")

print("\n✅ Fine!")