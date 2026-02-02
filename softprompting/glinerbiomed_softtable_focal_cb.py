# -*- coding: utf-8 -*-
"""
Training IBRIDATO:
- Architettura & Salvataggio: Identici al 'File 2' (Soft Embedding Table + Projection, NO MLP).
- Strategia Loss: Identica al 'File 1' (Class Balanced Weights + Focal Loss Gamma 4.0).
- Test finale sul test set dopo il training
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json
import torch
import torch.nn.functional as F
import time
from datetime import datetime
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, classification_report

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 15

# LEARNING RATES (Differenziati come richiesto dalla strategia avanzata)
LR_EMBED = 5e-4   # Learning rate per la tabella Soft Embeddings
LR_PROJ = 1e-4    # Learning rate per la Proiezione

WEIGHT_DECAY = 0.01
TEMPERATURE = 0.2
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
EARLY_STOPPING_PATIENCE = 4
RANDOM_SEED = 42

# PARAMETRI STRATEGIA AVANZATA (Class Balanced + Focal)
GAMMA_FOCAL_LOSS = 4.0
CB_BETA = 0.9999

# ==========================================
# KAGGLE / LOCAL PATHS
# ==========================================
if is_running_on_kaggle():
    # Modifica qui il nome del dataset Kaggle se necessario
    path = "/kaggle/input/jnlpa-6-2k5-1-2-complete/" 
    # Modifica qui il path del modello su Kaggle se necessario
    MODEL_NAME = "/kaggle/input/glinerbismall2/" 
else:
    path = "../dataset/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

DATASET_PATH = path + "dataset_tknlvl_bi.json"
VAL_DATASET_PATH = path + "val_dataset_tknlvl_bi.json"
TEST_DATASET_PATH = path + "test_dataset_tknlvl_bi.json"

LABEL2DESC_PATH = path + "label2desc.json"
LABEL2ID_PATH = path + "label2id.json"

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
# 3️⃣ DATASET & VALIDATION UTILS
# ==========================================================
def truncate_tokens_safe(tokens, tokenizer, max_len=None):
    if max_len is None: max_len = tokenizer.model_max_length
    if len(tokens) <= max_len: return tokens
    if tokens[0] == tokenizer.cls_token and tokens[-1] == tokenizer.sep_token and max_len >= 2:
        return [tokens[0]] + tokens[1:max_len-1] + [tokens[-1]]
    return tokens[:max_len]

def validate(val_records, txt_enc, txt_tok, soft_table, proj, device):
    """Validation Loop - Soft Table Mode (same logic as test_jnlpa_dual_mode.py)"""
    print("🔍 Validation (Soft Table Mode)...")
    txt_enc.eval()
    soft_table.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        # Calcolo Label Matrix da Soft Table
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1) # [Num_Labels, Proj_Dim]
        
        for record in val_records:
            tokens = record["tokens"]
            labels = record["labels"]
            
            if len(tokens) != len(labels): continue
            
            input_ids = txt_tok.convert_tokens_to_ids(tokens)
            
            max_len = getattr(txt_tok, "model_max_length", 512)
            if len(input_ids) > max_len:
                input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
                labels = labels[:len(input_ids)]
            
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            
            out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            logits = torch.matmul(H, label_matrix.T).squeeze(0) # [Seq_Len, Num_Labels]
            preds = logits.argmax(-1).cpu().numpy()
            
            for pred, true_label in zip(preds, labels):
                if true_label != -100:
                    y_true_all.append(true_label)
                    y_pred_all.append(pred)
                    
    # Metriche
    macro_f1 = precision_recall_fscore_support(y_true_all, y_pred_all, average="macro", zero_division=0)[2]
    micro_f1 = precision_recall_fscore_support(y_true_all, y_pred_all, average="micro", zero_division=0)[2]
    
    return macro_f1, micro_f1

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

print(f"📥 Caricamento Training Set: {DATASET_PATH}")
ds = TokenJsonDataset(DATASET_PATH, txt_tok)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

# Caricamento Validation Set (Raw per consistenza con Test script)
print(f"📥 Caricamento Validation Set: {VAL_DATASET_PATH}")
with open(VAL_DATASET_PATH, "r", encoding="utf-8") as f:
    val_records = json.load(f)
print(f"✅ Val Set caricato: {len(val_records)} records")

# Caricamento Test Set
print(f"📥 Caricamento Test Set: {TEST_DATASET_PATH}")
with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
    test_records = json.load(f)
print(f"✅ Test Set caricato: {len(test_records)} records")

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
        self.best_score = -float('inf') # Maximizing F1
        self.early_stop = False
    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
            return False # Not improved

early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

# ==========================================================
# 5️⃣ TRAINING LOOP
# ==========================================================
total_start_time = time.time()
print(f"\n🚀 Inizio Training | Embed LR: {LR_EMBED} | Proj LR: {LR_PROJ}")
print(f"🎯 Strategia: Focal (Gamma {GAMMA_FOCAL_LOSS}) + CB Weights (Beta {CB_BETA})")

txt_enc.eval()
soft_table.train()
proj.train()

best_val_f1 = -float('inf')
best_epoch = 0

# 🆕 Determina il nome del file UNA VOLTA con datetime
os.makedirs("savings", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
best_checkpoint_path = f"savings/soft_model_{dataset_name}_{timestamp}.pt"

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
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
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    # VALIDATION
    val_macro_f1, val_micro_f1 = validate(val_records, txt_enc, txt_tok, soft_table, proj, DEVICE)
    print(f"Epoch {epoch} | Val Macro F1: {val_macro_f1:.4f} | Val Micro F1: {val_micro_f1:.4f}")
    
    # Check Early Stopping & Saving Best
    is_improved = early_stopping(val_macro_f1)
    
    if is_improved:
        best_val_f1 = val_macro_f1
        best_epoch = epoch
        
        # 💾 SALVATAGGIO - Sovrascrive sempre lo stesso file
        save_dict = {
            # Components for inference
            'soft_embeddings': soft_table.state_dict(),
            'projection': proj.state_dict(),
            
            # Label info
            'label2id': label2id,
            'id2label': id2label,
            'label2desc': label2desc,
            
            # Metadata Standard (Metriche e Timestamp)
            'training_info': {
                'best_epoch': best_epoch,
                'best_val_macro_f1': best_val_f1,
                'timestamp': timestamp
            },
            
            # Tutti gli Iperparametri e Configurazioni
            'hyperparameters': {
                'lr_embed': LR_EMBED,
                'lr_proj': LR_PROJ,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'weight_decay': WEIGHT_DECAY,
                'temperature': TEMPERATURE,
                'grad_clip': GRAD_CLIP,
                'warmup_steps': WARMUP_STEPS,
                'es_patience': EARLY_STOPPING_PATIENCE,
                'gamma_focal': GAMMA_FOCAL_LOSS,
                'cb_beta': CB_BETA,
                'random_seed': RANDOM_SEED,
                'dataset_path': os.path.abspath(DATASET_PATH),
                'val_dataset_path': os.path.abspath(VAL_DATASET_PATH),
                'dataset_size': len(ds),
                'model_name': MODEL_NAME
            }
        }
        
        torch.save(save_dict, best_checkpoint_path)
        print(f"  💾 Best model saved (Macro F1: {best_val_f1:.4f}) to {best_checkpoint_path}")

    if early_stopping.early_stop:
        print(f"🛑 Early stopping at epoch {epoch}")
        break

total_training_time = time.time() - total_start_time
print(f"\n✅ Training Completato. Best Val Macro F1: {best_val_f1:.4f} (Epoch {best_epoch})")
print(f"⏱️  TEMPO TOTALE: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")

# ==========================================================
# 6️⃣ TEST FINALE SUL TEST SET
# ==========================================================
print("\n" + "="*70)
print("🧪 TEST FINALE SUL TEST SET")
print("="*70)

# Caricamento best checkpoint
print(f"📦 Caricamento best checkpoint: {best_checkpoint_path}")
checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
soft_table.load_state_dict(checkpoint['soft_embeddings'])
proj.load_state_dict(checkpoint['projection'])
print("✅ Best checkpoint caricato")

# Impostazione modalità eval
txt_enc.eval()
soft_table.eval()
proj.eval()

y_true_all = []
y_pred_all = []
n_skipped = 0

print(f"🔍 Processando {len(test_records)} esempi del test set...")

with torch.no_grad():
    # Calcolo Label Matrix da Soft Table
    label_vecs = proj(soft_table.weight)
    label_matrix = F.normalize(label_vecs, dim=-1)
    
    for idx, record in enumerate(test_records):
        if (idx + 1) % 100 == 0:
            print(f"   Processati {idx + 1}/{len(test_records)} esempi...")
        
        tokens = record["tokens"]
        labels = record["labels"]
        
        if len(tokens) != len(labels):
            n_skipped += 1
            continue
        
        input_ids = txt_tok.convert_tokens_to_ids(tokens)
        
        max_len = getattr(txt_tok, "model_max_length", 512)
        if len(input_ids) > max_len:
            input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
            labels = labels[:len(input_ids)]
        
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)
        
        out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
        H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        logits = torch.matmul(H, label_matrix.T).squeeze(0)
        preds = logits.argmax(-1).cpu().numpy()
        
        for pred, true_label in zip(preds, labels):
            if true_label != -100:
                y_true_all.append(true_label)
                y_pred_all.append(pred)

if n_skipped > 0:
    print(f"⚠️  Skipped {n_skipped} records (token/label mismatch)")

# 🆕 Calcolo SOLO Macro e Micro F1
all_label_ids = list(range(num_labels))

_, _, f1_macro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="macro", zero_division=0, labels=all_label_ids
)
_, _, f1_micro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="micro", zero_division=0, labels=all_label_ids
)

# 🆕 Stampa Risultati SEMPLIFICATA
print(f"\n{'='*70}")
print(f"📊 RISULTATI TEST FINALE")
print(f"{'='*70}")
print(f"Token valutati: {len(y_true_all):,}")
print(f"\n🎯 Macro F1:  {f1_macro:.4f}")
print(f"🎯 Micro F1:  {f1_micro:.4f}")
print(f"{'='*70}")

# Salvataggio Risultati
results_dir = "test_results"
os.makedirs(results_dir, exist_ok=True)

filename = f"{results_dir}/final_test_results_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test Finale - SOFT TABLE MODE\n\n")
    f.write(f"**Timestamp:** {timestamp}\n")
    f.write(f"**Environment:** {'Kaggle' if is_running_on_kaggle() else 'Local'}\n\n")
    
    f.write(f"## Training Info\n")
    f.write(f"- **Best Epoch:** {best_epoch}\n")
    f.write(f"- **Best Val Macro F1:** {best_val_f1:.4f}\n")
    f.write(f"- **Total Training Time:** {total_training_time:.1f}s ({total_training_time/60:.1f}min)\n")
    f.write(f"- **Gamma (Focal Loss):** {GAMMA_FOCAL_LOSS}\n")
    f.write(f"- **Beta (CB Weights):** {CB_BETA}\n\n")
    
    f.write(f"## Test Set Performance\n")
    f.write(f"- **Token valutati:** {len(y_true_all):,}\n")
    f.write(f"- **Macro F1:** {f1_macro:.4f}\n")
    f.write(f"- **Micro F1:** {f1_micro:.4f}\n\n")
    
    f.write(f"## Model Path\n")
    f.write(f"- **Checkpoint:** `{best_checkpoint_path}`\n")

print(f"\n💾 Risultati salvati in: {filename}")
print(f"✅ TRAINING E TEST COMPLETATI!")