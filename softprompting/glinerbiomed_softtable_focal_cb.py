# -*- coding: utf-8 -*-
"""
Training IBRIDATO:
- Architettura & Salvataggio: Identici al 'File 2' (Soft Embedding Table + Projection, NO MLP).
- Strategia Loss: Identica al 'File 1' (Class Balanced Weights + Focal Loss Gamma 4.0).
"""

import json
import torch
import torch.nn.functional as F
import time
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10

# LEARNING RATES (Differenziati come richiesto dalla strategia avanzata)
LR_EMBED = 1e-3   # Learning rate per la tabella Soft Embeddings
LR_PROJ = 1e-3    # Learning rate per la Proiezione

WEIGHT_DECAY = 0.01
TEMPERATURE = 1.0
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42

# PARAMETRI STRATEGIA AVANZATA (Class Balanced + Focal)
GAMMA_FOCAL_LOSS = 4.0
CB_BETA = 0.9999

DATASET_PATH = "../dataset/dataset_tokenlevel_balanced.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ FOCAL LOSS & CLASS BALANCED WEIGHTS
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        
        # Calcolo base Focal
        focal_weight = (1 - p_t) ** self.gamma
        
        # Applicazione Alpha (Class Balanced Weights)
        if self.alpha is not None:
            valid_mask = targets != self.ignore_index
            alpha_t = torch.ones_like(targets, dtype=logits.dtype)
            # alpha ha dimensione [num_classes], targets ha dimensione [batch*seq]
            # Assicuriamoci che alpha sia sullo stesso device
            alpha_tensor = self.alpha.to(targets.device)
            alpha_t[valid_mask] = alpha_tensor[targets[valid_mask]]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        # Ritorna la media sui token validi
        mask = targets != self.ignore_index
        if mask.sum() == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return focal_loss[mask].mean()

def get_cb_weights(dataset_path, label2id, beta=0.9999):
    """Calcola i pesi 'Class Balanced' (Cui et al. 2019)"""
    print("⚖️  Calcolo pesi Class Balanced...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    label_counts = Counter()
    for record in data:
        for label_id in record["labels"]:
            if label_id != -100:
                label_counts[label_id] += 1
                
    num_classes = len(label2id)
    weights = torch.ones(num_classes)
    
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else:
            weights[label_id] = 0.0 
        
    # Normalizzazione: somma pesi = numero classi
    weights = weights / weights.sum() * num_classes
    return weights

# ==========================================================
# 2️⃣ SETUP MODELLO (Stile File 2 - NO MLP)
# ==========================================================
print("📦 Caricamento modello base...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# FREEZING
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = False
# La proiezione DEVE essere trainabile
for p in proj.parameters(): p.requires_grad = True

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# --- HARD EMBEDDINGS & SOFT TABLE ---
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
label_names = [id2label[i] for i in range(num_labels)]

print("⚙️  Generazione Hard Embeddings (Base)...")
desc_texts = [label2desc[name] for name in label_names]
batch_desc = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

with torch.no_grad():
    out = lbl_enc(**batch_desc)
    mask = batch_desc["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
HE_VECTORS = pooled

print(f"✅ Hard Embeddings generati: {HE_VECTORS.shape}")

# Creazione Tabella Soft Embeddings (Inizializzata con HE)
soft_table = nn.Embedding(num_labels, HE_VECTORS.size(1))
with torch.no_grad():
    soft_table.weight.copy_(HE_VECTORS)

# Registra nel modello (opzionale, ma utile per compatibilità GLiNER)
core.token_rep_layer.soft_label_embeddings = soft_table
soft_table.to(DEVICE)

# ==========================================================
# 3️⃣ DATASET
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer):
        with open(path_json, "r", encoding="utf-8") as f: self.records = json.load(f)
        self.tok = tokenizer
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(rec["labels"]),
        }

def collate_batch(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

ds = TokenJsonDataset(DATASET_PATH, txt_tok)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

# ==========================================================
# 4️⃣ TRAINING SETUP (Strategia File 1 applicata a File 2)
# ==========================================================

# 1. Calcolo Pesi e Loss
cb_weights = get_cb_weights(DATASET_PATH, label2id, beta=CB_BETA).to(DEVICE)
criterion = FocalLoss(alpha=cb_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

# 2. Optimizer con Learning Rates differenziati
# Qui applichiamo la strategia LR differenziata agli oggetti del File 2
optimizer = optim.AdamW([
    {"params": soft_table.parameters(), "lr": LR_EMBED},
    {"params": proj.parameters(), "lr": LR_PROJ}
], weight_decay=WEIGHT_DECAY)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader)*EPOCHS)

# 3. Early Stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

# ==========================================================
# 5️⃣ TRAINING LOOP
# ==========================================================
print(f"\n🚀 Inizio Training | Embed LR: {LR_EMBED} | Proj LR: {LR_PROJ}")
print(f"🎯 Strategia: Focal (Gamma {GAMMA_FOCAL_LOSS}) + CB Weights (Beta {CB_BETA})")

txt_enc.eval()
soft_table.train()
proj.train()

best_loss = float('inf')
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        
        # 1. Text Embeddings (Frozen)
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        # 2. Label Embeddings (Soft Table diretta)
        # soft_table.weight è [num_labels, hidden]. 
        # Proiettiamo -> [num_labels, proj_dim]
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        # 3. Similarità e Loss
        logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
        loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(soft_table.parameters()) + list(proj.parameters()), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    # Check Early Stopping & Saving Best
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch
        
        # 💾 SALVATAGGIO IDENTICO AL FILE 2
        # Questo crea il dizionario che il tuo script di test si aspetta (se configurato per soft prompts)
        save_dict = {
            'soft_embeddings': soft_table.state_dict(),
            'projection': proj.state_dict(),
            'label2id': label2id,
            'id2label': id2label,
            'config': {
                'gamma': GAMMA_FOCAL_LOSS,
                'beta': CB_BETA
            }
        }
        os.makedirs("savings", exist_ok=True)
        save_path = "savings/soft_embeddings_best_hybrid.pt"
        torch.save(save_dict, save_path)
        print(f"  💾 Best model saved to {save_path}")

    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print(f"🛑 Early stopping at epoch {epoch}")
        break

print(f"\n✅ Training Completato. Best Loss: {best_loss:.4f} (Epoch {best_epoch})")

# ==========================================================
# 🧪 TEST RAPIDO INFERENCE
# ==========================================================
print("\n🧪 Test Inference Rapido")
test_txt = "Aspirin inhibits NF-kappa B activation."
inp = txt_tok(test_txt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    soft_table.eval()
    proj.eval()
    
    H = F.normalize(txt_enc(**inp).last_hidden_state, dim=-1)
    L = F.normalize(proj(soft_table.weight), dim=-1)
    
    logits = torch.matmul(H, L.T)
    preds = logits.argmax(-1).squeeze().cpu().tolist()
    tokens = txt_tok.convert_ids_to_tokens(inp["input_ids"][0])
    
    for t, p in zip(tokens, preds):
        print(f"{t:<15} -> {id2label[p]}")