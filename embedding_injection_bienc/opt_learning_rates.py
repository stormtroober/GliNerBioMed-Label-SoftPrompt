# -*- coding: utf-8 -*-
import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import optuna
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from gliner import GLiNER

# ==========================================
# 🔧 CONFIGURAZIONE FISSA (Dai tuoi risultati precedenti)
# ==========================================
PROMPT_LEN = 32
POOLING_MODE = "conv1d"

BATCH_SIZE = 64
EPOCHS = 6
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.011641058260782156
GAMMA_FOCAL = 5.0
CB_BETA = 0.9999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALIDATION_RATIO = 0.1

def is_running_on_kaggle(): return os.path.exists('/kaggle/input')
input_dir = "/kaggle/input/standard15000/" if is_running_on_kaggle() else "../dataset/"
DATASET_PATH = input_dir + "dataset_tokenlevel_simple.json"
LABEL2DESC_PATH = input_dir + "label2desc.json"
LABEL2ID_PATH = input_dir + "label2id.json"
MODEL_NAME = '/kaggle/input/glinerbismall2/' if is_running_on_kaggle() else "Ihor/gliner-biomed-bi-small-v1.0"

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER & POOLER
# ==========================================================
class PromptPooler(nn.Module):
    def __init__(self, embed_dim, prompt_len, mode="adaptive_avg", max_seq_len=512):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        
        if mode == "adaptive_avg":
            self.pooler = nn.AdaptiveAvgPool1d(prompt_len)
        elif mode == "adaptive_max":
            self.pooler = nn.AdaptiveMaxPool1d(prompt_len)
        elif mode == "attention":
            self.queries = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d":
            # Conv1D downsampling: apprende come comprimere la sequenza
            # Usiamo kernel_size e stride calcolati dinamicamente nel forward,
            # ma qui definiamo i layer convoluzionali
            self.conv_layers = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d_strided":
            # Versione più aggressiva con stride learnable
            # Usa una singola convoluzione con kernel grande per catturare contesto
            self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    
    def forward(self, x, attention_mask=None):
        B, seq_len, dim = x.shape
        
        if self.mode in ["adaptive_avg", "adaptive_max"]:
            x_t = x.transpose(1, 2)  # (B, dim, seq_len)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).float()
                if self.mode == "adaptive_avg":
                    x_t = x_t * mask_expanded
                else:
                    x_t = x_t.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = self.pooler(x_t)
            return pooled.transpose(1, 2)  # (B, prompt_len, dim)
        
        elif self.mode == "attention":
            queries = self.queries.expand(B, -1, -1)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)
        
        elif self.mode == "conv1d":
            # Applica mask prima della convoluzione
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            
            x_t = x.transpose(1, 2)  # (B, dim, seq_len)
            conv_out = self.conv_layers(x_t)  # (B, dim, seq_len)
            pooled = self.adaptive_pool(conv_out)  # (B, dim, prompt_len)
            pooled = pooled.transpose(1, 2)  # (B, prompt_len, dim)
            return self.norm(pooled)
        
        elif self.mode == "conv1d_strided":
            # Versione con gating mechanism
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            
            x_t = x.transpose(1, 2)  # (B, dim, seq_len)
            conv_out = self.conv(x_t)  # (B, dim, seq_len)
            pooled = self.adaptive_pool(conv_out)  # (B, dim, prompt_len)
            pooled = pooled.transpose(1, 2)  # (B, prompt_len, dim)
            
            # Gating: permette al modello di decidere quanto "fidarsi" della convoluzione
            gate = self.gate(pooled)
            pooled = pooled * gate
            
            return self.norm(pooled)

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg",
                 max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        if hidden_dim is None: hidden_dim = embed_dim * 4
            
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        self.pooler = None
        if prompt_len is not None:
            self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode, max_seq_len=max_seq_len)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        if self.pooler is not None:
            x = self.pooler(x, attention_mask)
        return x

# ==========================================================
# 2️⃣ LOSS FUNCTION
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha 

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss) 
        mask = target != self.ignore_index
        if self.alpha is not None:
            alpha_t = self.alpha[target[mask]]
            focal_loss = alpha_t * (1 - pt[mask]) ** self.gamma * ce_loss[mask]
        else:
            focal_loss = (1 - pt[mask]) ** self.gamma * ce_loss[mask]
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# ... Load Data & Model ...
print("📦 Loading Backbone...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model
txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = False
# Reset proj state later
initial_proj_state = {k: v.clone() for k, v in proj.state_dict().items()}

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# Setup Dataset & Weights (Fixed for this run)
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
# ... (Codice caricamento descrizioni identico al precedente) ...
label_names = [k for k in label2id.keys()] # Simplification
desc_texts = [label2desc[name] for name in label_names]
desc_inputs = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

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

# Pre-calculate weights (FISSI per questo esperimento)
def get_cb_weights_simple(path, l2id, dev, beta):
    with open(path) as f: data = json.load(f)
    cnt = Counter([l for r in data for l in r["labels"] if l != -100])
    weights = torch.ones(len(l2id)).to(dev)
    for l_id in l2id.values():
        c = cnt.get(l_id, 0)
        weights[l_id] = (1-beta)/(1-beta**c) if c > 0 else 0
    return weights / weights.sum() * len(l2id)

class_weights = get_cb_weights_simple(DATASET_PATH, label2id, DEVICE, CB_BETA)

# Load FULL dataset
full_ds = TokenJsonDataset(DATASET_PATH, txt_tok)
dataset_size = len(full_ds)

# SPLIT TRAIN/VALIDATION
val_size = int(dataset_size * VALIDATION_RATIO)
train_size = dataset_size - val_size

# Si usa un generator con seed fisso per riproducibilità dello split
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

# LOADERS
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

print(f"🔪 Dataset Split: {len(train_ds)} Train | {len(val_ds)} Val")

# ==========================================
# 🎯 OPTUNA OBJECTIVE: LEARNING RATES
# ==========================================
def objective(trial):
    # 1. SUGGEST PARAMETERS
    # LR MLP: Cerca tra 1e-5 e 1e-2 in scala logaritmica
    lr_mlp = trial.suggest_float("lr_mlp", 1e-5, 1e-2, log=True)
    
    # STRATEGIA PROJECTION LR: 
    # Optuna decide se disaccoppiare gli LR o tenerli uguali
    uncouple = trial.suggest_categorical("uncouple_lr", [True, False])
    
    if uncouple:
        # Se disaccoppiati, cerca un LR specifico per la projection
        lr_proj = trial.suggest_float("lr_proj", 1e-5, 1e-2, log=True)
    else:
        # Altrimenti usa lo stesso
        lr_proj = lr_mlp

    print(f"\n🧪 TRIAL {trial.number} | LR MLP: {lr_mlp:.2e} | LR PROJ: {lr_proj:.2e}")

    # 2. SETUP MODEL
    proj.load_state_dict(initial_proj_state)
    proj.train()
    
    model_encoder = MLPPromptEncoder(
        lbl_enc.embeddings.word_embeddings, 
        vocab_size=lbl_enc.embeddings.word_embeddings.num_embeddings,
        embed_dim=lbl_enc.embeddings.word_embeddings.embedding_dim,
        prompt_len=PROMPT_LEN, 
        pooling_mode=POOLING_MODE,
        max_seq_len=desc_input_ids.shape[1]
    ).to(DEVICE)
    model_encoder.train()
    
    # 3. OPTIMIZER
    optimizer = optim.AdamW([
        {"params": model_encoder.parameters(), "lr": lr_mlp},
        {"params": proj.parameters(), "lr": lr_proj}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(len(train_loader)*EPOCHS*0.1), 
                                                num_training_steps=len(train_loader)*EPOCHS)
    
    loss_fn = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL)

    # 4. TRAINING LOOP
    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model_encoder.train()
        proj.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            
            with torch.no_grad():
                H_text = F.normalize(txt_enc(batch["input_ids"], batch["attention_mask"]).last_hidden_state, dim=-1)
            
            # Encoder
            desc_embeds = model_encoder(desc_input_ids, desc_attn_mask)
            pooled_mask = torch.ones(desc_embeds.shape[:2], device=DEVICE).long() if PROMPT_LEN else desc_attn_mask
            
            # Label Enc
            lbl_out = lbl_enc(inputs_embeds=desc_embeds, attention_mask=pooled_mask).last_hidden_state
            mask_exp = pooled_mask.unsqueeze(-1).float()
            # Mean pooling manuale per sicurezza
            pooled = (lbl_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            H_label = F.normalize(proj(pooled), dim=-1)
            
            logits = torch.matmul(H_text, H_label.T) / TEMPERATURE
            loss = loss_fn(logits.view(-1, len(label2id)), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # --- VALIDATION (End of Epoch) ---
        model_encoder.eval()
        proj.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k,v in batch.items()}
                
                H_text = F.normalize(txt_enc(batch["input_ids"], batch["attention_mask"]).last_hidden_state, dim=-1)
                
                desc_embeds = model_encoder(desc_input_ids, desc_attn_mask)
                pooled_mask = torch.ones(desc_embeds.shape[:2], device=DEVICE).long() if PROMPT_LEN else desc_attn_mask
                
                lbl_out = lbl_enc(inputs_embeds=desc_embeds, attention_mask=pooled_mask).last_hidden_state
                mask_exp = pooled_mask.unsqueeze(-1).float()
                pooled = (lbl_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
                H_label = F.normalize(proj(pooled), dim=-1)
                
                logits = torch.matmul(H_text, H_label.T) / TEMPERATURE
                
                # PREDICTIONS
                preds = torch.argmax(logits, dim=-1)
                
                # Masking pad/ignore (-100)
                active_loss = batch["labels"].view(-1) != -100
                active_labels = batch["labels"].view(-1)[active_loss]
                active_preds = preds.view(-1)[active_loss]
                
                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
        
        # CALCOLO MICRO F1 (Più sensibile per ranking generale)
        # O Macro F1 se preferisci bilanciamento classi
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Report & Pruning su F1 (Maximize)
        trial.report(val_f1, epoch)
        if trial.should_prune(): 
            raise optuna.TrialPruned()
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
    return best_val_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(f"🏆 BEST PARAMS: {study.best_params}")