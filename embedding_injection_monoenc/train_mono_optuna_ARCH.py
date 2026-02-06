# -*- coding: utf-8 -*-
"""
Training Mono-Encoder - OPTUNA ARCHITECTURE SEARCH
- Obiettivo: Trovare la migliore architettura del PromptEncoder
- Parametri: Prompt Length, Pooling Mode, Hidden Ratio, Dropout
- Training veloce (5 epoche) per valutare la convergenza strutturale
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
# 🔧 CONFIGURAZIONE BASE (Fissa per Architecture Search)
# ==========================================================
BATCH_SIZE = 8
EPOCHS = 5  # Epoche ridotte per architecture search
LR_MLP = 1e-4 # LR ragionevole medio per valutare architetture
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
RANDOM_SEED = 42

# Parametri Fissi Loss
GAMMA_FOCAL_LOSS = 5.0
CB_BETA = 0.9999
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
    TRAIN_PATH = input_dir + "dataset/dataset_tknlvl_mono.json" # MODIFICATO!
    VAL_PATH = input_dir + "dataset/val_dataset_tknlvl_mono.json" # MODIFICATO!
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = "urchade/gliner_small-v2.1"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER (Dinamica)
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

print(f"📏 Caricamento Datasets da: \n Train:{TRAIN_PATH} \n Val:{VAL_PATH}")

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
        # Cleanup CLS/SEP
        if len(input_ids) > 0 and input_ids[0] == self.tok.cls_token_id:
            input_ids = input_ids[1:]
            labels = labels[1:]
        if len(input_ids) > 0 and input_ids[-1] == self.tok.sep_token_id:
            input_ids = input_ids[:-1]
            labels = labels[:-1]
        # Truncation
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

# ==========================================================
# 🎯 OPTUNA OBJECTIVE (ARCHITETTURA)
# ==========================================================
def objective(trial):
    # --- 1. SPUNTA PARAMETRI ARCHITETTURALI ---
    
    # Lunghezza Prompt: Valutiamo corti, medi e lunghi
    prompt_len = trial.suggest_categorical("prompt_len", [16, 32, 64])
    
    # Pooling Strategy: Come comprimere il prompt
    pooling_mode = trial.suggest_categorical("pooling_mode", ["adaptive_avg", "attention", "conv1d"])
    
    # Complessità MLP interno: Rapporto hidden dim (es. 2x o 4x embed_dim)
    hidden_dim_ratio = trial.suggest_int("hidden_dim_ratio", 2, 8, step=2)
    
    # Dropout (Regolarizzazione strutturale)
    dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
    
    print(f"\n🧪 TRIAL {trial.number} | Struct: Len={prompt_len}, Pool={pooling_mode}, Ratio={hidden_dim_ratio}x, Drop={dropout}")

    # --- 2. CONFIGURAZIONE DINAMICA DEGLI INPUT ---
    # Max Text Length dipende dalla Prompt Length scelta!
    TOTAL_PROMPT_TOKENS = num_labels * prompt_len
    MAX_MODEL_LEN = 512
    MAX_TEXT_LEN = MAX_MODEL_LEN - TOTAL_PROMPT_TOKENS - 5 
    
    if MAX_TEXT_LEN < 128: 
        raise optuna.TrialPruned(f"Text len too small: {MAX_TEXT_LEN}")

    # --- 3. DATALOADERS (Ricreati per cambiare max_len se necessario) ---
    train_ds = TokenJsonDataset(TRAIN_PATH, tokenizer, max_len=MAX_TEXT_LEN)
    val_ds = TokenJsonDataset(VAL_PATH, tokenizer, max_len=MAX_TEXT_LEN)
    
    # Piccolo subsample per velocizzare Architecture Search? 
    # No, usiamo tutto ma poche epoche (5). 
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id))

    # Desc Inputs
    desc_texts = [label2desc[name] for name in label_names]
    desc_inputs = tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True)
    desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
    desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

    # --- 4. LOSS & MODEL ---
    class_weights = get_cb_weights(TRAIN_PATH, label2id, DEVICE, beta=CB_BETA)
    ce_loss_fn = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

    prompt_encoder = MLPPromptEncoder(
        original_word_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=dropout, # Ottimizzato
        prompt_len=prompt_len, # Ottimizzato
        pooling_mode=pooling_mode, # Ottimizzato
        hidden_dim_ratio=hidden_dim_ratio, # Ottimizzato
    ).to(DEVICE)
    
    optimizer = optim.AdamW(prompt_encoder.parameters(), lr=LR_MLP, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    best_loss = float('inf')
    
    # --- 5. TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        prompt_encoder.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            # Flatten & Expand (come in train_mono.py) 
            # Qui la dimensione dipende da prompt_len corrente
            soft_prompts_flat = soft_prompts.view(-1, embed_dim) 
            prompts_len_flat = soft_prompts_flat.shape[0] # NumLabels * CurrentPromptLen
            
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
            
            # Extract Indices (Dynamic based on prompt_len)
            text_start = 1 + prompts_len_flat + 1
            text_end = text_start + batch["input_ids"].shape[1]
            text_reps = sequence_output[:, text_start:text_end, :] 
            
            prompt_reps_seq = sequence_output[:, 1:1+prompts_len_flat, :] 
            prompt_reps_reshaped = prompt_reps_seq.view(B_batch, num_labels, prompt_len, embed_dim)
            prompt_vectors = prompt_reps_reshaped.mean(dim=2) # Mean over prompt len
            
            H_text = F.normalize(text_reps, dim=-1)
            H_prompts = F.normalize(prompt_vectors, dim=-1) 
            
            logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / TEMPERATURE
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
                prompt_reps_reshaped = prompt_reps_seq.view(B_batch, num_labels, prompt_len, embed_dim)
                prompt_vectors = prompt_reps_reshaped.mean(dim=2) 
                
                H_text = F.normalize(text_reps, dim=-1)
                H_prompts = F.normalize(prompt_vectors, dim=-1) 
                
                logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / TEMPERATURE
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
    print(f"\n🏗️ Inizio Architecture Search (30 Trials, 5 Epoche)")
    
    # Pruning aggressivo (salva tempo su architetture scarse)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=30)
    
    print("\n🏆 BEST ARCHITECTURE:")
    print(study.best_params)
    print(f"Best Val Loss: {study.best_value}")

    # SAVE RESULTS
    os.makedirs("softprompting/optunas/mono_arch", exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "softprompting/optunas/mono_arch"

    # JSON Results
    results = {
        "timestamp": timestamp,
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
        
    with open(os.path.join(output_dir, f"arch_results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # PLOTS
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Parallel Coordinate (Params overview)
        # Semplificato custom plot
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        vals = [t.value for t in complete_trials]
        
        # Prompt Len vs Loss
        lens = [t.params['prompt_len'] for t in complete_trials]
        axes[0,0].boxplot([[v for l,v in zip(lens, vals) if l==x] for x in [16,32,64]], labels=[16,32,64])
        axes[0,0].set_title("Prompt Length Impact")
        axes[0,0].set_ylabel("Val Loss")

        # Pooling Mode vs Loss
        modes = sorted(list(set([t.params['pooling_mode'] for t in complete_trials])))
        data_modes = [[v for m,v in zip([t.params['pooling_mode'] for t in complete_trials], vals) if m==x] for x in modes]
        axes[0,1].boxplot(data_modes, labels=modes)
        axes[0,1].set_title("Pooling Strategy Impact")
        
        # Dropout vs Loss (Scatter)
        drops = [t.params['dropout'] for t in complete_trials]
        axes[1,0].scatter(drops, vals, c='blue', alpha=0.6)
        axes[1,0].set_title("Dropout Correlation")
        axes[1,0].set_xlabel("Dropout Rate")
        axes[1,0].set_ylabel("Val Loss")

        # Hidden Ratio vs Loss
        ratios = [t.params['hidden_dim_ratio'] for t in complete_trials]
        axes[1,1].scatter(ratios, vals, c='green', alpha=0.6)
        axes[1,1].set_title("MLP Hidden Size Ratio")
        
        plt.suptitle(f"Mono-Encoder Architecture Search - {timestamp}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"arch_summary_{timestamp}.png"))
        print("📊 Grafici salvati.")
        
    except Exception as e:
        print(f"Errore plotting: {e}")
