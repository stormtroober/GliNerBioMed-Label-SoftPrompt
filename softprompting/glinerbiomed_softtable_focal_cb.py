# -*- coding: utf-8 -*-
"""
Training IBRIDATO:
- Architettura & Salvataggio: Identici al 'File 2' (Soft Embedding Table + Projection, NO MLP).
- Strategia Loss: Identica al 'File 1' (Class Balanced Weights + Focal Loss Gamma 4.0).
- Kaggle Ready: supporto per Kaggle/Local paths, validation separata/split, logging e saving migliorato.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json
import torch
import torch.nn.functional as F
import time
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
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
VAL_SPLIT_RATIO = 0.2  # Usato solo se USE_SEPARATE_VAL_FILE = False

# 🔄 Flag per gestione validation:
# - True: usa file di validation separato (es. dataset_anatEM ha val_dataset_tknlvl_bi.json)
# - False: split del training set (80% train, 20% val)
USE_SEPARATE_VAL_FILE = True

# PARAMETRI STRATEGIA AVANZATA (Class Balanced + Focal)
GAMMA_FOCAL_LOSS = 4.0
CB_BETA = 0.9999

# ==========================================
# KAGGLE / LOCAL PATHS
# ==========================================
if is_running_on_kaggle():
    path = "/kaggle/input/bc5dr-full/"
    #path = "/kaggle/input/tknlvl-jnlpa-5k/"
    MODEL_NAME = "/kaggle/input/glinerbismall2/"
else:
    path = "../dataset/"
    path_val = "../dataset_anatEM/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

DATASET_PATH = path + "dataset_tknlvl_bi.json"
LABEL2DESC_PATH = path + "label2desc.json"
LABEL2ID_PATH = path + "label2id.json"

# Path validation (usato solo se USE_SEPARATE_VAL_FILE = True)
if USE_SEPARATE_VAL_FILE:
    if is_running_on_kaggle():
        VAL_DATASET_PATH = "/kaggle/input/bc5dr-full/val_dataset_tknlvl_bi.json"
    else:
        VAL_DATASET_PATH = "../dataset_anatEM/val_dataset_tknlvl_bi.json"

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
    def __init__(self, path_json, tokenizer, label2id=None):
        with open(path_json, "r", encoding="utf-8") as f: self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(rec["labels"]),
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

print("📚 Caricamento dataset...")
ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)
print(f"📊 Total dataset size: {len(ds)}\n")

# ==========================================================
# 4️⃣ TRAINING SETUP (Strategia File 1 applicata a File 2)
# ==========================================================

# 1. Calcolo Pesi e Loss
cb_weights = get_cb_weights(DATASET_PATH, label2id, beta=CB_BETA).to(DEVICE)
criterion = FocalLoss(alpha=cb_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

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
            print(f"⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# ✨ Funzione per eseguire un'epoca di training
def run_epoch(loader, soft_table, proj, txt_enc, criterion, optimizer, scheduler=None):
    """Esegue un'epoca di training"""
    soft_table.train()
    proj.train()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 1. Text Embeddings (Frozen)
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        # 2. Label Embeddings (Soft Table diretta)
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        # 3. Similarità e Loss
        logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
        loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(soft_table.parameters()) + list(proj.parameters()), GRAD_CLIP)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        total_acc += (preds[mask] == batch["labels"][mask]).float().sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = (total_acc / n_tokens) * 100
    return avg_loss, avg_acc

# ✨ Funzione per eseguire validation (senza gradient)
@torch.no_grad()
def run_validation_epoch(loader, soft_table, proj, txt_enc, criterion):
    """Esegue validation (no gradient)"""
    soft_table.eval()
    proj.eval()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
        loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        total_acc += (preds[mask] == batch["labels"][mask]).float().sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = (total_acc / n_tokens) * 100
    return avg_loss, avg_acc

# ✨ Timer totale
total_start_time = time.time()

# ==========================================================
# 5️⃣ TRAINING LOOP
# ==========================================================
print(f"\n🚀 Inizio Training | Embed LR: {LR_EMBED} | Proj LR: {LR_PROJ}")
print(f"🎯 Strategia: Focal (Gamma {GAMMA_FOCAL_LOSS}) + CB Weights (Beta {CB_BETA})")

txt_enc.eval()

# ✨ DataLoader per training e validation
if USE_SEPARATE_VAL_FILE:
    # Usa file di validation separato
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    
    print("📚 Caricamento validation dataset da file separato...")
    val_ds = TokenJsonDataset(VAL_DATASET_PATH, txt_tok, label2id)
    print(f"📊 Validation dataset size: {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
else:
    # Split del training set
    print(f"� Splitting training set ({int((1-VAL_SPLIT_RATIO)*100)}% train, {int(VAL_SPLIT_RATIO*100)}% val)...")
    val_size = int(len(ds) * VAL_SPLIT_RATIO)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    print(f"📊 Training size: {len(train_ds)}, Validation size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))

# 2. Optimizer con Learning Rates differenziati
optimizer = optim.AdamW([
    {"params": soft_table.parameters(), "lr": LR_EMBED},
    {"params": proj.parameters(), "lr": LR_PROJ}
], weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

training_start_time = time.time()

# ✨ Inizializza early stopping
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
best_loss = float('inf')
best_epoch = 0

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    # Training
    train_loss, train_acc = run_epoch(
        tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"),
        soft_table, proj, txt_enc, criterion, optimizer, scheduler
    )
    
    # ✨ Validation
    val_loss, val_acc = run_validation_epoch(
        tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False),
        soft_table, proj, txt_enc, criterion
    )
    
    epoch_time = time.time() - epoch_start_time
    
    # ✨ Traccia miglior epoca basata su validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        
        # 💾 SALVATAGGIO BEST MODEL
        save_dict_best = {
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
        best_save_path = "savings/soft_embeddings_best_hybrid.pt"
        torch.save(save_dict_best, best_save_path)
        print(f"  💾 Best model saved to {best_save_path}")
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | lr={current_lr:.2e} | time={epoch_time:.1f}s")
    
    # ✨ Early stopping check basato su validation loss
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\n🛑 Early stopping triggered at epoch {epoch}")
        print(f"🏆 Best val_loss: {best_loss:.4f} (epoch {best_epoch})")
        break

# ✨ Timer totale
total_training_time = time.time() - total_start_time

print(f"\n⏱️  TEMPO TOTALE: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
print(f"🏆 Best val_loss: {best_loss:.4f} (epoch {best_epoch})")

# ==========================================================
# 💾 SALVATAGGIO MODELLO FINALE
# ==========================================================
print(f"\n💾 Salvataggio modello finale...")

os.makedirs("savings", exist_ok=True)

# Estrai nome dataset dal path
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
dataset_size = len(ds)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Nome file con dataset info e timestamp (parametri salvati dentro il .pt)
save_path = f"savings/softprompt_{dataset_name}_size{dataset_size}_{timestamp}.pt"

torch.save({
    # State dicts
    'soft_embeddings': soft_table.state_dict(),
    'projection': proj.state_dict(),
    
    # Dataset info
    'dataset_name': dataset_name,
    'dataset_path': DATASET_PATH,
    'dataset_size': dataset_size,
    'use_separate_val_file': USE_SEPARATE_VAL_FILE,
    'val_dataset_path': VAL_DATASET_PATH if USE_SEPARATE_VAL_FILE else None,
    'val_split_ratio': None if USE_SEPARATE_VAL_FILE else VAL_SPLIT_RATIO,
    'val_dataset_size': len(val_ds),
    
    # Tutti gli iperparametri
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr_embed': LR_EMBED,
        'lr_proj': LR_PROJ,
        'weight_decay': WEIGHT_DECAY,
        'temperature': TEMPERATURE,
        'grad_clip': GRAD_CLIP,
        'warmup_steps': WARMUP_STEPS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'random_seed': RANDOM_SEED,
        'gamma_focal_loss': GAMMA_FOCAL_LOSS,
        'cb_beta': CB_BETA,
    },
    
    # Training info
    'training_info': {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'total_training_time_seconds': total_training_time,
        'model_name': MODEL_NAME,
    },
    
    # Label mappings (utili per inference)
    'label2id': label2id,
    'label2desc': label2desc,
    'id2label': id2label,
    
    # Config specifico per soft prompting
    'config': {
        'gamma': GAMMA_FOCAL_LOSS,
        'beta': CB_BETA
    }
}, save_path)

print(f"✅ Modello salvato: {save_path}")
print(f"📊 Dataset: {dataset_name} (size: {dataset_size})")
print(f"📋 Parametri salvati nel checkpoint")

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