# -*- coding: utf-8 -*-

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import optuna
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 🔧 CONFIGURAZIONE GLOBALE
# ==========================================================
TRAIN_PROJECTION = True
BATCH_SIZE = 64
EPOCHS = 8  # Riduciamo leggermente per fare più trial veloci con Optuna

# Parametri Fissi (Non ottimizzati in questo giro)
LR_MLP = 1e-3
LR_PROJ = 1e-3
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
GRAD_CLIP = 1.0
WARMUP_PERCENTAGE = 0.15
RANDOM_SEED = 42
DROPOUT_RATE = 0.1
# POOLING_MODE rimosso da qui - ora è ottimizzato da Optuna

# Configurazione Loss
GAMMA_FOCAL_LOSS = 4.5
CB_BETA = 0.9999
WEIGHT_STRATEGY = "ClassBalanced"

# Paths
input_dir = "/kaggle/input/standard5000/" if is_running_on_kaggle() else ""
DATASET_PATH = input_dir + "dataset_tokenlevel_simple.json"
LABEL2DESC_PATH = input_dir + "label2desc.json"
LABEL2ID_PATH = input_dir + "label2id.json"

if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/glinerbismall2/' 
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

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

# ==========================================================
# 3️⃣ CARICAMENTO DATI E MODELLO (Eseguito una volta sola)
# ==========================================================
print("📦 Caricamento modello Base...")
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
for p in proj.parameters(): p.requires_grad = TRAIN_PROJECTION

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# IMPORTANTE: Salviamo lo stato iniziale della Projection per resettarla ad ogni trial di Optuna
initial_proj_state = {k: v.clone() for k, v in proj.state_dict().items()}

# Info Embeddings per Encoder Dinamico
original_word_embeddings = lbl_enc.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

# --- DATASET & PESI ---
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)

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
            weights[label_id] = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / weights[label_id]
        else: weights[label_id] = 0.0 
    weights = weights / weights.sum() * num_classes
    return weights

# Prepare DataLoader
ds = TokenJsonDataset(DATASET_PATH, txt_tok)
class_weights = get_cb_weights(DATASET_PATH, label2id, DEVICE, beta=CB_BETA)
ce_loss = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

# ==========================================================
# 4️⃣ OPTUNA OBJECTIVE FUNCTION
# ==========================================================

def objective(trial):
    # 1. SUGGEST PARAMETERS
    # Usiamo 0 per indicare "None" (ovvero lunghezza originale, senza pooling)
    p_len_candidate = trial.suggest_categorical("prompt_len", [8, 16, 32, 64, 128])
    current_prompt_len = None if p_len_candidate == 0 else p_len_candidate
    
    # Ottimizza anche il pooling mode - ora include conv1d e conv1d_strided
    current_pooling_mode = trial.suggest_categorical("pooling_mode", [
        "adaptive_avg", 
        # "adaptive_max",
        "attention", 
        "conv1d",
        "conv1d_strided"
    ])
    
    print(f"\n🧪 TRIAL {trial.number} | Prompt Len: {current_prompt_len if current_prompt_len else 'ORIGINAL'} | Pooling: {current_pooling_mode}")

    # 2. RESET PROJECTION (Per Fairness)
    # Ricarichiamo i pesi originali della projection per non partire avvantaggiati
    if TRAIN_PROJECTION:
        proj.load_state_dict(initial_proj_state)
        proj.train()
    else:
        proj.eval()

    # 3. BUILD DYNAMIC ENCODER
    # Creiamo un nuovo encoder specifico per questo trial
    model_encoder = MLPPromptEncoder(
        original_word_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=DROPOUT_RATE,
        prompt_len=current_prompt_len, 
        pooling_mode=current_pooling_mode,
        max_seq_len=desc_input_ids.shape[1]  # Passa la lunghezza massima delle descrizioni
    ).to(DEVICE)
    model_encoder.train()

    # 4. OPTIMIZER SETUP
    optimizer_grouped_parameters = [{"params": model_encoder.parameters(), "lr": LR_MLP}]
    if TRAIN_PROJECTION:
        optimizer_grouped_parameters.append({"params": proj.parameters(), "lr": LR_PROJ})

    import math
    dataset_size = len(ds)
    WARMUP_STEPS = round(math.ceil(dataset_size / BATCH_SIZE) * EPOCHS * WARMUP_PERCENTAGE)
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader)*EPOCHS)

    # 5. TRAINING LOOP (Senza TQDM interno per pulizia)
    txt_enc.eval() 
    lbl_enc.eval()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with torch.no_grad():
                out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            # Encoder Forward
            soft_embeds = model_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            
            # Mask Handling
            if current_prompt_len is not None:
                pooled_attn_mask = torch.ones(soft_embeds.shape[0], soft_embeds.shape[1], dtype=torch.long, device=DEVICE)
            else:
                pooled_attn_mask = desc_attn_mask
            
            # Backbone Label Forward
            outputs = lbl_enc(inputs_embeds=soft_embeds, attention_mask=pooled_attn_mask)
            
            # Final Pooling & Proj
            mask_exp = pooled_attn_mask.unsqueeze(-1).float()
            pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            label_matrix = F.normalize(proj(pooled), dim=-1)
            
            logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
            loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 6. OPTUNA REPORTING & PRUNING
        # Comunica a Optuna la loss attuale
        trial.report(avg_loss, epoch)
        
        # Se la loss è troppo alta rispetto agli altri trial, interrompi subito
        if trial.should_prune():
            print(f"✂️ Trial Pruned at Epoch {epoch} (Loss: {avg_loss:.4f})")
            raise optuna.TrialPruned()
            
        print(f"   Epoch {epoch}/{EPOCHS} -> Loss: {avg_loss:.4f}")

    return avg_loss

# ==========================================================
# 5️⃣ MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    print("\n🔍 Avvio ottimizzazione Optuna per PROMPT LENGTH e POOLING MODE...")
    
    # Crea lo studio: direction="minimize" perché vogliamo minimizzare la Loss
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)
    )
    
    # Lancia l'ottimizzazione
    study.optimize(objective, n_trials=30)

    print("\n" + "="*40)
    print("🏆 RISULTATI OPTUNA")
    print("="*40)
    print(f"Best Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    best_len = study.best_params['prompt_len']
    final_len = best_len if best_len != 0 else "None (Original)"
    best_pooling = study.best_params['pooling_mode']
    
    print(f"\n✅ I vincitori sono:")
    print(f"   - Prompt Length = {final_len}")
    print(f"   - Pooling Mode = {best_pooling}")
    print("Ora puoi modificare le variabili PROMPT_LEN e POOLING_MODE in alto nel file con questi valori e rieseguire un training lungo per salvare il modello.")

#     = 128
#    - Pooling Mode = adaptive_avg