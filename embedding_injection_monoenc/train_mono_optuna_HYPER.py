# -*- coding: utf-8 -*-
"""
Training Mono-Encoder - OPTUNA HYPERPARAMETER SEARCH
- Obiettivo: Trovare i migliori iperparametri di training per l'architettura scelta
- Architettura Fissata (da studio precedente):
    - Prompt Len: 32 (CORRECTED from arch study)
    - Pooling: conv1d
    - Hidden Dim Ratio: 4
    - Dropout: 0.1 (CORRECTED from arch study)
- Parametri Ottimizzati: LR, Temperature, Gamma Focal Loss, Beta Class Balanced
"""

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import time
import gc
import optuna
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm
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
# 🔧 CONFIGURAZIONE BASE (Fissa)
# ==========================================================
BATCH_SIZE = 8
EPOCHS = 10  
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
RANDOM_SEED = 42

# --- ARCHITETTURA FISSATA (Da Optuna ARCH) ---
FIXED_PROMPT_LEN = 32
FIXED_POOLING_MODE = "conv1d"
FIXED_HIDDEN_RATIO = 4
FIXED_DROPOUT = 0.1
# ---------------------------------------------

WEIGHT_STRATEGY = "ClassBalanced"

# Paths
if is_running_on_kaggle():
    input_dir = "/kaggle/input/standard16600/"
    TRAIN_PATH = input_dir + "dataset_tknlvl_mono.json"
    VAL_PATH = input_dir + "val_dataset_tknlvl_mono.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = '/kaggle/input/gliner2-1small/'
else:
    input_dir = "../" 
    TRAIN_PATH = input_dir + "dataset/dataset_tknlvl_mono.json"
    VAL_PATH = input_dir + "dataset/val_dataset_tknlvl_mono.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = "urchade/gliner_small-v2.1"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER (Identica a ARCH)
# ==========================================================
class PromptPooler(nn.Module):
    def __init__(self, embed_dim, prompt_len, mode="adaptive_avg"):
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

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim_ratio=4, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        hidden_dim = int(embed_dim * hidden_dim_ratio)
            
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
            self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode)

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
# 3️⃣ PREPARAZIONE MODELLO BASE
# ==========================================================
print("📦 Caricamento modello Base...")
model_wrapper = GLiNER.from_pretrained(MODEL_NAME)
model = model_wrapper.model
tokenizer = model_wrapper.data_processor.transformer_tokenizer
backbone = model.token_rep_layer.bert_layer.model
original_word_embeddings = backbone.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

for p in backbone.parameters(): p.requires_grad = False
backbone.to(DEVICE)
backbone.eval()
print(f"✅ Backbone Frozen. Dim: {embed_dim}")

# ==========================================================
# 4️⃣ CARICAMENTO DATASET
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)

# Calcolo MAX LEN basato sui parametri FISSATI
TOTAL_PROMPT_TOKENS = num_labels * FIXED_PROMPT_LEN
MAX_MODEL_LEN = 512
MAX_TEXT_LEN = MAX_MODEL_LEN - TOTAL_PROMPT_TOKENS - 5 
if MAX_TEXT_LEN < 100:
    print(f"⚠️ ATTENZIONE: MAX_TEXT_LEN è molto basso ({MAX_TEXT_LEN})")

print(f"📏 Max Text Len: {MAX_TEXT_LEN} (Prompt: {TOTAL_PROMPT_TOKENS})")

class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, max_len=512):
        with open(path_json, "r", encoding="utf-8") as f: self.records = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        labels = rec["labels"]
        if len(input_ids) > 0 and input_ids[0] == self.tok.cls_token_id:
            input_ids = input_ids[1:]
            labels = labels[1:]
        if len(input_ids) > 0 and input_ids[-1] == self.tok.sep_token_id:
            input_ids = input_ids[:-1]
            labels = labels[:-1]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(labels),
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

# Pre-loading Datasets (qui è statico, non cambia per trial)
print("📊 Loading Datasets...")
train_ds = TokenJsonDataset(TRAIN_PATH, tokenizer, max_len=MAX_TEXT_LEN)
val_ds = TokenJsonDataset(VAL_PATH, tokenizer, max_len=MAX_TEXT_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id))

desc_texts = [label2desc[name] for name in label_names]
desc_inputs = tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

# ==========================================================
# 🎯 OPTUNA OBJECTIVE (HYPERPARAMETERS)
# ==========================================================
def objective(trial):
    # --- 1. SPUNTA PARAMETRI TRAINING ---
    
    # LR: Esplorazione logaritmica (spesso il parametro più critico)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # Temperature: Scaling dei logits (impatta la sharpness delle predizioni)
    temperature = trial.suggest_float("temperature", 0.01, 0.2)
    
    # Focal Gamma: Quanto punire gli esempi difficili (2.0 è standard, ma 5.0 funz benissimo in Bio)
    gamma = trial.suggest_float("gamma", 2.0, 6.0)
    
    # Class Balanced Beta: 0.999 (soft) o 0.9999 (harder rebalancing)
    beta = trial.suggest_categorical("beta", [0.999, 0.9999])
    
    print(f"\n🧪 TRIAL {trial.number} | LR={lr:.2e}, T={temperature:.3f}, G={gamma:.1f}, B={beta}")

    # --- 2. CONFIG LOSS & MODEL ---
    class_weights = get_cb_weights(TRAIN_PATH, label2id, DEVICE, beta=beta)
    ce_loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, ignore_index=-100)

    prompt_encoder = MLPPromptEncoder(
        original_word_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=FIXED_DROPOUT,
        prompt_len=FIXED_PROMPT_LEN,
        pooling_mode=FIXED_POOLING_MODE,
        hidden_dim_ratio=FIXED_HIDDEN_RATIO,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(prompt_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    best_loss = float('inf')
    
    # --- 3. TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        prompt_encoder.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            soft_prompts_flat = soft_prompts.view(-1, embed_dim) 
            prompts_len_flat = soft_prompts_flat.shape[0]
            
            batch_soft_prompts = soft_prompts_flat.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
            
            text_embeds = backbone.embeddings(batch["input_ids"])
            cls_token = torch.tensor([[tokenizer.cls_token_id]] * batch["input_ids"].shape[0], device=DEVICE)
            sep_token = torch.tensor([[tokenizer.sep_token_id]] * batch["input_ids"].shape[0], device=DEVICE)
            cls_embed = backbone.embeddings(cls_token) # (B, 1, D)
            sep_embed = backbone.embeddings(sep_token) # (B, 1, D)
            
            inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
            
            B_batch = batch["input_ids"].shape[0]
            prompt_mask = torch.ones((B_batch, prompts_len_flat), device=DEVICE)
            extra = torch.ones((B_batch, 1), device=DEVICE)
            full_mask = torch.cat([extra, prompt_mask, extra, batch["attention_mask"], extra], dim=1)
            
            outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)) 
            sequence_output = outputs.last_hidden_state
            
            text_start = 1 + prompts_len_flat + 1
            text_end = text_start + batch["input_ids"].shape[1]
            text_reps = sequence_output[:, text_start:text_end, :] 
            
            prompt_reps_seq = sequence_output[:, 1:1+prompts_len_flat, :] 
            prompt_reps_reshaped = prompt_reps_seq.view(B_batch, num_labels, FIXED_PROMPT_LEN, embed_dim)
            prompt_vectors = prompt_reps_reshaped.mean(dim=2) 
            
            H_text = F.normalize(text_reps, dim=-1)
            H_prompts = F.normalize(prompt_vectors, dim=-1) 
            
            logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / temperature
            loss = ce_loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # VALIDATION
        prompt_encoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                
                soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
                soft_prompts_flat = soft_prompts.view(-1, embed_dim)
                prompts_len_flat = soft_prompts_flat.shape[0]
                batch_soft_prompts = soft_prompts_flat.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
                
                text_embeds = backbone.embeddings(batch["input_ids"])
                cls_token = torch.tensor([[tokenizer.cls_token_id]] * batch["input_ids"].shape[0], device=DEVICE)
                sep_token = torch.tensor([[tokenizer.sep_token_id]] * batch["input_ids"].shape[0], device=DEVICE)
                cls_embed = backbone.embeddings(cls_token)
                sep_embed = backbone.embeddings(sep_token)
                
                inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
                
                B_batch = batch["input_ids"].shape[0]
                prompt_mask = torch.ones((B_batch, prompts_len_flat), device=DEVICE)
                extra = torch.ones((B_batch, 1), device=DEVICE)
                full_mask = torch.cat([extra, prompt_mask, extra, batch["attention_mask"], extra], dim=1)
                
                outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)) 
                sequence_output = outputs.last_hidden_state
                
                text_start = 1 + prompts_len_flat + 1
                text_end = text_start + batch["input_ids"].shape[1]
                text_reps = sequence_output[:, text_start:text_end, :] 
                
                prompt_reps_seq = sequence_output[:, 1:1+prompts_len_flat, :] 
                prompt_reps_reshaped = prompt_reps_seq.view(B_batch, num_labels, FIXED_PROMPT_LEN, embed_dim)
                prompt_vectors = prompt_reps_reshaped.mean(dim=2) 
                
                H_text = F.normalize(text_reps, dim=-1)
                H_prompts = F.normalize(prompt_vectors, dim=-1) 
                
                logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / temperature
                loss = ce_loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        if avg_val_loss < best_loss: best_loss = avg_val_loss
        print(f"  Ep {epoch} | T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f}")

    return best_loss

# ==========================================================
# 🚀 MAIN
# ==========================================================
if __name__ == "__main__":
    print(f"\n🏗️ Inizio Hyperparam Search (20 Trials, 10 Epoche)")
    print(f"🔒 Fixed Arch: Len={FIXED_PROMPT_LEN}, Pool={FIXED_POOLING_MODE}, Ratio={FIXED_HIDDEN_RATIO}, Drop={FIXED_DROPOUT}")
    
    # Pruning standard
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=20)
    
    print("\n🏆 BEST PARAMS:")
    print(study.best_params)
    print(f"Best Val Loss: {study.best_value}")

    # SAVE RESULTS
    os.makedirs("softprompting/optunas/mono_hyper", exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "softprompting/optunas/mono_hyper"

    # JSON Results
    results = {
        "timestamp": timestamp,
        "fixed_architecture": {
            "prompt_len": FIXED_PROMPT_LEN,
            "pooling_mode": FIXED_POOLING_MODE,
            "hidden_dim_ratio": FIXED_HIDDEN_RATIO,
            "dropout": FIXED_DROPOUT
        },
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": []
    }
    for t in study.trials:
        results["trials"].append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state)
        })
        
    with open(os.path.join(output_dir, f"hyper_results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # PLOTS
    try:
        # Importanza Iperparametri
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title("Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hyper_importance_{timestamp}.png"))
        plt.close()

        # Slice Plot (LR vs Others)
        optuna.visualization.matplotlib.plot_slice(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hyper_slice_{timestamp}.png"))
        plt.close()
        
        print("📊 Grafici salvati.")
        
    except Exception as e:
        print(f"Errore plotting avanzato (richiede optuna visualization): {e}")
        # Fallback manual plot base
        lrs = [t.params['lr'] for t in study.trials if t.state==TrialState.COMPLETE]
        vals = [t.value for t in study.trials if t.state==TrialState.COMPLETE]
        plt.figure()
        plt.scatter(lrs, vals)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Val Loss')
        plt.title('LR Impact')
        plt.savefig(os.path.join(output_dir, f"manual_lr_plot_{timestamp}.png"))