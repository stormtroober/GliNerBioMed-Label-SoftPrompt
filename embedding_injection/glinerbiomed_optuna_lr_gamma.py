# -*- coding: utf-8 -*-

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import optuna
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 🔧 CONFIGURAZIONE BASE
# ==========================================================
TRAIN_PROJECTION = True
BATCH_SIZE = 128
EPOCHS = 15 # Reduced for Optuna speed (can be increased)
# LR and GAMMA will be optimized by Optuna

WEIGHT_DECAY = 0.01
TEMPERATURE = 0.011641058260782156
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1 # Fixed roughly based on previous
RANDOM_SEED = 42
DROPOUT_RATE = 0.1

PROMPT_LEN = 32
POOLING_MODE = "conv1d"

CB_BETA = 0.9999
WEIGHT_STRATEGY = "ClassBalanced"
VALIDATION_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5

input_dir = "/kaggle/input/standard15000/" if is_running_on_kaggle() else ""

DATASET_PATH = input_dir + "dataset_tokenlevel_simple.json"
LABEL2DESC_PATH = input_dir + "label2desc.json"
LABEL2ID_PATH = input_dir + "label2id.json"

if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/glinerbismall2/' 
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER (CON POOLING)
# ==========================================================
class PromptPooler(nn.Module):
    """Riduce la sequenza da (B, seq_len, dim) a (B, prompt_len, dim)"""
    
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
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)
        
        elif self.mode == "conv1d":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv_layers(x_t)
            pooled = self.adaptive_pool(conv_out)
            pooled = pooled.transpose(1, 2)
            return self.norm(pooled)
        
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
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================================
# 3️⃣ DATASET & HELPERS
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer):
        print(f"📖 Leggendo {path_json}...")
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
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_counts = Counter()
    for record in data:
        for label_id in record["labels"]:
            if label_id != -100:
                label_counts[label_id] += 1
    num_classes = len(label2id)
    weights = torch.ones(num_classes).to(device)
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else:
            weights[label_id] = 0.0 
    weights = weights / weights.sum() * num_classes
    return weights

# ==========================================================
# 4️⃣ PRE-LOADING (STATIC)
# ==========================================================
print("📦 Caricamento Tokenizers e Dati...")
# Per evitare di ricaricare il dataset a ogni trial, lo carichiamo qui.
# Tuttavia, per il Tokenizer serve il modello. 
# Carichiamo un modello dummy o parziale solo per i tokenizer se possibile, 
# oppure carichiamo tutto nel trial. 
# Dato che Model è "Ihor/gliner-biomed-bi-small-v1.0"
dummy_model = GLiNER.from_pretrained(MODEL_NAME)
enc_dummy = dummy_model.model.token_rep_layer.bert_layer.model
txt_tok = AutoTokenizer.from_pretrained(enc_dummy.config._name_or_path)

full_ds = TokenJsonDataset(DATASET_PATH, txt_tok)
dataset_size = len(full_ds)
val_size = int(dataset_size * VALIDATION_RATIO)
train_size = dataset_size - val_size
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

# Labels
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)
desc_texts = [label2desc[name] for name in label_names]

# CB Weights fissi per il dataset (li calcoliamo una volta)
class_weights_static = get_cb_weights(DATASET_PATH, label2id, DEVICE, beta=CB_BETA)

print(f"✅ Pre-loading completato. Ready per Optuna.")

# ==========================================================
# 5️⃣ OPTUNA OBJECTIVE
# ==========================================================
def objective(trial):
    # --- HYPERPARAMETERS TUNING ---
    # Focus su LR e GAMMA
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 8.0, step=0.5)
    
    # Parametri derivati
    LR_MLP = lr
    LR_PROJ = lr
    GAMMA_FOCAL_LOSS = gamma
    
    # --- MODEL SETUP (Fresh per every trial) ---
    model = GLiNER.from_pretrained(MODEL_NAME)
    core = model.model
    txt_enc = core.token_rep_layer.bert_layer.model
    lbl_enc = core.token_rep_layer.labels_encoder.model
    proj = core.token_rep_layer.labels_projection
    
    lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)
    
    # Freezing
    for p in txt_enc.parameters(): p.requires_grad = False
    for p in lbl_enc.parameters(): p.requires_grad = False
    for p in proj.parameters(): p.requires_grad = TRAIN_PROJECTION

    txt_enc.to(DEVICE)
    lbl_enc.to(DEVICE)
    proj.to(DEVICE)

    # Prompt Encoder Info
    original_word_embeddings = lbl_enc.embeddings.word_embeddings
    vocab_size = original_word_embeddings.num_embeddings
    embed_dim = original_word_embeddings.embedding_dim

    # Desc Embeddings
    desc_inputs = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True)
    desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
    desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

    # Prompt Encoder
    prompt_encoder = MLPPromptEncoder(
        original_word_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=DROPOUT_RATE,
        prompt_len=PROMPT_LEN,
        pooling_mode=POOLING_MODE,
        max_seq_len=desc_input_ids.shape[1]
    ).to(DEVICE)

    # Optimizer
    optimizer_grouped_parameters = [
        {"params": prompt_encoder.parameters(), "lr": LR_MLP}
    ]
    if TRAIN_PROJECTION:
        optimizer_grouped_parameters.append({"params": proj.parameters(), "lr": LR_PROJ})
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)

    # Loss
    ce_loss = FocalLoss(alpha=class_weights_static, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

    # Scheduler
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # --- TRAINING LOOP ---
    txt_enc.eval() 
    lbl_enc.eval()
    if TRAIN_PROJECTION: proj.train()
    else: proj.eval()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        prompt_encoder.train()
        if TRAIN_PROJECTION: proj.train()
        
        # Train
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with torch.no_grad():
                out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)

            soft_embeds = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            
            if PROMPT_LEN is not None:
                pooled_attn_mask = torch.ones(soft_embeds.shape[0], soft_embeds.shape[1], 
                                            dtype=torch.long, device=DEVICE)
            else:
                pooled_attn_mask = desc_attn_mask
            
            outputs = lbl_enc(inputs_embeds=soft_embeds, attention_mask=pooled_attn_mask)
            mask_exp = pooled_attn_mask.unsqueeze(-1).float()
            pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            label_matrix = F.normalize(proj(pooled), dim=-1)
            
            logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
            loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        # Validation
        prompt_encoder.eval()
        if TRAIN_PROJECTION: proj.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
                
                soft_embeds = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
                if PROMPT_LEN is not None:
                    pooled_attn_mask = torch.ones(soft_embeds.shape[0], soft_embeds.shape[1], 
                                                dtype=torch.long, device=DEVICE)
                else:
                    pooled_attn_mask = desc_attn_mask
                
                outputs = lbl_enc(inputs_embeds=soft_embeds, attention_mask=pooled_attn_mask)
                mask_exp = pooled_attn_mask.unsqueeze(-1).float()
                pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
                label_matrix = F.normalize(proj(pooled), dim=-1)
                
                logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
                v_loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Report to Optuna
        trial.report(avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    study_name = "optuna_gliner_lr_gamma"
    storage_name = "sqlite:///{}.db".format(study_name)
    
    # Se vuoi ricominciare da capo ogni volta:
    # try: os.remove(f"{study_name}.db")
    # except: pass

    study = optuna.create_study(
        study_name=study_name, 
        direction="minimize", 
        storage=storage_name, 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    print("\n🔍 Inizio Ottimizzazione Optuna...")
    study.optimize(objective, n_trials=30)  # Puoi aumentare n_trials
    
    print("\n🎉 Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Salva migliori parametri su file
    with open("best_hyperparameters.json", "w") as f:
        json.dump(trial.params, f, indent=4)
        print("💾 Parametri salvati in best_hyperparameters.json")
