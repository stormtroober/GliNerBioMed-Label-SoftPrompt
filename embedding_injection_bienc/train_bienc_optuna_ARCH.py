# -*- coding: utf-8 -*-
"""
Training Bi-Encoder - OPTUNA ARCHITECTURE SEARCH
- Obiettivo: Trovare la migliore architettura del PromptEncoder per il Biencoder
- Parametri Ottimizzati: Prompt Length, Pooling Mode
- Training veloce (10 epoche) per valutare la convergenza strutturale
- Architettura Biencoder: txt_enc (frozen) + lbl_enc (frozen) + projection (trainable)
"""

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import gc
import optuna
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from optuna.trial import TrialState

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

# 🧹 PULIZIA MEMORIA INIZIALE
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 🔧 CONFIGURAZIONE BASE (Fissa per Architecture Search)
# ==========================================================
TRAIN_PROJECTION = True
BATCH_SIZE = 64
EPOCHS = 10  # Epoche ridotte per architecture search
LR_MLP = 1e-3  # LR ragionevole medio per valutare architetture
LR_PROJ = 1e-3
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
RANDOM_SEED = 42

# Parametri Fissi Loss
GAMMA_FOCAL_LOSS = 5.0
CB_BETA = 0.9999
WEIGHT_STRATEGY = "ClassBalanced"

# Parametri Architetturali Fissi
DROPOUT = 0.1

# Paths
if is_running_on_kaggle():
    input_dir = "/kaggle/input/jnlpa-6-2k5-1-2-complete/"
    TRAIN_PATH = input_dir + "dataset_tknlvl_bi.json"
    VAL_PATH = input_dir + "val_dataset_tknlvl_bi.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = '/kaggle/input/glinerbismall2/'
else:
    input_dir = "../dataset/"
    TRAIN_PATH = input_dir + "dataset_tknlvl_bi.json"
    VAL_PATH = input_dir + "val_dataset_tknlvl_bi.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER (Dinamica)
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
            self.conv_layers = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d_strided":
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
            x_t = x.transpose(1, 2)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).float()
                if self.mode == "adaptive_avg":
                    x_t = x_t * mask_expanded
                else: 
                    x_t = x_t.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = self.pooler(x_t)
            return pooled.transpose(1, 2)
        
        elif self.mode == "attention":
            queries = self.queries.expand(B, -1, -1)
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)
            
        elif self.mode == "conv1d":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv_layers(x_t)
            pooled = self.adaptive_pool(conv_out)
            return self.norm(pooled.transpose(1, 2))

        elif self.mode == "conv1d_strided":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv(x_t)
            pooled = self.adaptive_pool(conv_out)
            pooled = pooled.transpose(1, 2)
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
        
        if hidden_dim is None: 
            hidden_dim = embed_dim * 4
            
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
# 2️⃣ FOCAL LOSS
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

# ==========================================================
# 3️⃣ PREPARAZIONE MODELLO BASE (Biencoder)
# ==========================================================
print("📦 Caricamento modello Biencoder Base...")
model_wrapper = GLiNER.from_pretrained(MODEL_NAME)
core = model_wrapper.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# 🔒 FREEZING Encoders
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = False
for p in proj.parameters(): p.requires_grad = TRAIN_PROJECTION

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# Salva stato iniziale della projection per reset ad ogni trial
initial_proj_state = {k: v.clone() for k, v in proj.state_dict().items()}

# Info Embeddings per Encoder Dinamico
original_word_embeddings = lbl_enc.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

print(f"✅ Biencoder Backbone Frozen. Embed Dim: {embed_dim}")

# ==========================================================
# 4️⃣ CARICAMENTO DATASET
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)

desc_texts = [label2desc[name] for name in label_names]
desc_inputs = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

print(f"📏 Caricamento Datasets da:\n TRAIN: {TRAIN_PATH}\n VAL: {VAL_PATH}")

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

def get_cb_weights(dataset_path, label2id, device, beta=0.9999):
    with open(dataset_path, "r", encoding="utf-8") as f: data = json.load(f)
    label_counts = Counter()
    for record in data:
        for label_id in record["labels"]:
            if label_id != -100: label_counts[label_id] += 1
    num_classes = len(label2id)
    weights = torch.ones(num_classes).to(device)
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else: weights[label_id] = 0.0 
    weights = weights / weights.sum() * num_classes
    return weights

# Pre-load datasets (separate files, no split)
print("📊 Loading Datasets...")
train_ds = TokenJsonDataset(TRAIN_PATH, txt_tok)
val_ds = TokenJsonDataset(VAL_PATH, txt_tok)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

print(f"🔪 Dataset Sizes: Train={len(train_ds)} | Val={len(val_ds)}")

class_weights = get_cb_weights(TRAIN_PATH, label2id, DEVICE, beta=CB_BETA)
ce_loss_fn = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

# ==========================================================
# 🎯 OPTUNA OBJECTIVE (ARCHITETTURA BIENCODER)
# ==========================================================
def objective(trial):
    # --- 1. SPUNTA PARAMETRI ARCHITETTURALI ---
    
    # Lunghezza Prompt: Valutiamo corti, medi e lunghi
    prompt_len = trial.suggest_categorical("prompt_len", [16, 32, 64])
    
    # Pooling Strategy: Come comprimere il prompt
    pooling_mode = trial.suggest_categorical("pooling_mode", ["adaptive_avg", "attention", "conv1d"])
    
    print(f"\n🧪 TRIAL {trial.number} | Struct: Len={prompt_len}, Pool={pooling_mode} | Fixed: Drop={DROPOUT}")

    # --- 2. RESET PROJECTION (Per Fairness) ---
    if TRAIN_PROJECTION:
        proj.load_state_dict(initial_proj_state)
        proj.train()
    else:
        proj.eval()

    # --- 3. BUILD DYNAMIC ENCODER ---
    prompt_encoder = MLPPromptEncoder(
        original_word_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=DROPOUT,
        prompt_len=prompt_len,
        pooling_mode=pooling_mode,
        max_seq_len=desc_input_ids.shape[1]
    ).to(DEVICE)

    # --- 4. OPTIMIZER ---
    optimizer_grouped_parameters = [{"params": prompt_encoder.parameters(), "lr": LR_MLP}]
    if TRAIN_PROJECTION:
        optimizer_grouped_parameters.append({"params": proj.parameters(), "lr": LR_PROJ})
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    best_val_loss = float('inf')
    
    # Timing tracking
    epoch_times = []
    
    # --- 5. TRAINING LOOP ---
    txt_enc.eval()
    lbl_enc.eval()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"\n⏰ Epoch {epoch}/{EPOCHS} - Start: {datetime.now().strftime('%H:%M:%S')}")
        prompt_encoder.train()
        if TRAIN_PROJECTION: proj.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            # Text encoding (frozen)
            with torch.no_grad():
                out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            # Prompt Encoder forward
            soft_embeds = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            
            # [CLS] + soft_embeds + [SEP]
            num_labels_batch = soft_embeds.shape[0]
            cls_embeds = lbl_enc.embeddings.word_embeddings(
                torch.tensor([[lbl_tok.cls_token_id]], device=DEVICE)
            ).expand(num_labels_batch, -1, -1)
            sep_embeds = lbl_enc.embeddings.word_embeddings(
                torch.tensor([[lbl_tok.sep_token_id]], device=DEVICE)
            ).expand(num_labels_batch, -1, -1)
            
            enclosed_soft_embeds = torch.cat([cls_embeds, soft_embeds, sep_embeds], dim=1)
            total_len = 1 + soft_embeds.shape[1] + 1
            pooled_attn_mask = torch.ones(num_labels_batch, total_len, dtype=torch.long, device=DEVICE)
            
            # Label encoder forward (frozen)
            outputs = lbl_enc(inputs_embeds=enclosed_soft_embeds, attention_mask=pooled_attn_mask)
            
            # Mean pooling + Projection
            mask_exp = pooled_attn_mask.unsqueeze(-1).float()
            pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            label_matrix = F.normalize(proj(pooled), dim=-1)
            
            # Logits & Loss
            logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
            loss = ce_loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
            if TRAIN_PROJECTION:
                torch.nn.utils.clip_grad_norm_(proj.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # VALIDATION
        prompt_encoder.eval()
        if TRAIN_PROJECTION: proj.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                
                out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
                
                soft_embeds = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
                
                num_labels_batch = soft_embeds.shape[0]
                cls_embeds = lbl_enc.embeddings.word_embeddings(
                    torch.tensor([[lbl_tok.cls_token_id]], device=DEVICE)
                ).expand(num_labels_batch, -1, -1)
                sep_embeds = lbl_enc.embeddings.word_embeddings(
                    torch.tensor([[lbl_tok.sep_token_id]], device=DEVICE)
                ).expand(num_labels_batch, -1, -1)
                
                enclosed_soft_embeds = torch.cat([cls_embeds, soft_embeds, sep_embeds], dim=1)
                total_len = 1 + soft_embeds.shape[1] + 1
                pooled_attn_mask = torch.ones(num_labels_batch, total_len, dtype=torch.long, device=DEVICE)
                
                outputs = lbl_enc(inputs_embeds=enclosed_soft_embeds, attention_mask=pooled_attn_mask)
                
                mask_exp = pooled_attn_mask.unsqueeze(-1).float()
                pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
                label_matrix = F.normalize(proj(pooled), dim=-1)
                
                logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
                loss = ce_loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        if avg_val_loss < best_val_loss: best_val_loss = avg_val_loss
        
        # Epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        print(f"  Ep {epoch} | T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f}")
        print(f"⏰ Epoch {epoch}/{EPOCHS} - End: {datetime.now().strftime('%H:%M:%S')} | Duration: {epoch_duration:.2f}s")

    # Final timing report
    if epoch_times:
        total_time = sum(epoch_times)
        avg_time = total_time / len(epoch_times)
        print(f"\n📊 TIMING REPORT (Trial {trial.number}):")
        print(f"   Total Training Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"   Average Epoch Time: {avg_time:.2f}s")
        print(f"   Min Epoch Time: {min(epoch_times):.2f}s")
        print(f"   Max Epoch Time: {max(epoch_times):.2f}s")
    
    return best_val_loss

# ==========================================================
# 🚀 MAIN
# ==========================================================
if __name__ == "__main__":
    print(f"\n🏗️ Inizio Biencoder Architecture Search (9 Combinazioni Totali, {EPOCHS} Epoche)")
    print("📐 Grid Search: 3 prompt_len × 3 pooling_mode = 9 trials esatti")
    
    # Grid Sampler: prova tutte le combinazioni esattamente una volta
    search_space = {
        "prompt_len": [16, 32, 64],
        "pooling_mode": ["adaptive_avg", "attention", "conv1d"]
    }
    
    # Pruner meno aggressivo: permette almeno 6 epoche prima di prunare
    # e richiede almeno 3 trial completati prima di iniziare a prunare
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,  # Aspetta almeno 3 trial prima di prunare
        n_warmup_steps=6,    # Aspetta almeno 6 epoche prima di prunare
        interval_steps=1
    )
    
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(search_space),
        pruner=pruner
    )
    
    # n_trials=9 è esattamente il numero di combinazioni nel grid
    study.optimize(objective, n_trials=9)
    
    print("\n🏆 BEST ARCHITECTURE:")
    print(study.best_params)
    print(f"Best Val Loss: {study.best_value}")

    # SAVE RESULTS
    os.makedirs("softprompting/optunas/bienc_arch", exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "softprompting/optunas/bienc_arch"

    # JSON Results
    results = {
        "timestamp": timestamp,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "fixed_config": {
            "train_projection": TRAIN_PROJECTION,
            "lr_mlp": LR_MLP,
            "lr_proj": LR_PROJ,
            "temperature": TEMPERATURE,
            "gamma_focal_loss": GAMMA_FOCAL_LOSS,
            "cb_beta": CB_BETA,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        "trials": []
    }
    for t in study.trials:
        results["trials"].append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state)
        })
        
    with open(os.path.join(output_dir, f"arch_results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # PLOTS
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        vals = [t.value for t in complete_trials]
        
        # Prompt Len vs Loss
        lens = [t.params['prompt_len'] for t in complete_trials]
        prompt_len_options = sorted(list(set(lens)))
        axes[0].boxplot([[v for l,v in zip(lens, vals) if l==x] for x in prompt_len_options], labels=prompt_len_options)
        axes[0].set_title("Prompt Length Impact")
        axes[0].set_ylabel("Val Loss")

        # Pooling Mode vs Loss
        modes = sorted(list(set([t.params['pooling_mode'] for t in complete_trials])))
        data_modes = [[v for m,v in zip([t.params['pooling_mode'] for t in complete_trials], vals) if m==x] for x in modes]
        axes[1].boxplot(data_modes, labels=modes)
        axes[1].set_title("Pooling Strategy Impact")
        
        plt.suptitle(f"Bi-Encoder Architecture Search - {timestamp}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"arch_summary_{timestamp}.png"))
        print("📊 Grafici salvati.")
        
    except Exception as e:
        print(f"Errore plotting: {e}")
