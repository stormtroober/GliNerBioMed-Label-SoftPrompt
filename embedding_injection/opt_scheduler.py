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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from gliner import GLiNER

# ==========================================
# 🔧 PARAMETRI FISSI (Dalle ricerche precedenti)
# ==========================================
PROMPT_LEN = 32         
POOLING_MODE = "conv1d" 
FIXED_LR_MLP = 0.002     
FIXED_LR_PROJ = 0.002    

# Loss Params Fixed
CB_BETA = 0.9999
FOCAL_GAMMA = 5.0
TEMP = 0.0116

BATCH_SIZE = 64
EPOCHS = 6
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def is_running_on_kaggle(): return os.path.exists('/kaggle/input')
input_dir = "/kaggle/input/standard5000/" if is_running_on_kaggle() else ""
DATASET_PATH = input_dir + "../dataset/dataset_tokenlevel_simple.json"
LABEL2DESC_PATH = input_dir + "../label2desc.json"
LABEL2ID_PATH = input_dir + "../label2id.json"
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
        print(f"🔧 FocalLoss initialized with Gamma={gamma}")

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

# ==========================================================
# 4️⃣ LOAD MODEL & DATA
# ==========================================================
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
initial_proj_state = {k: v.clone() for k, v in proj.state_dict().items()}

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# Load labels
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)

label_names = [k for k in label2id.keys()]
desc_texts = [label2desc[name] for name in label_names]
desc_inputs = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

# Load dataset
ds = TokenJsonDataset(DATASET_PATH, txt_tok)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

# Pre-calcoliamo i count
with open(DATASET_PATH) as f: raw_data = json.load(f)
global_counts = Counter([l for r in raw_data for l in r["labels"] if l != -100])

# Funzione Helper per ricalcolare pesi al volo
def get_weights_dynamic(label_counts, num_classes, beta, device):
    weights = torch.ones(num_classes).to(device)
    for lid, count in label_counts.items():
        if count > 0:
            eff_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[lid] = 1.0 / eff_num
        else:
            weights[lid] = 0.0
    return weights / weights.sum() * num_classes

# ==========================================
# 🎯 OPTUNA OBJECTIVE: SCHEDULER
# ==========================================
def objective(trial):
    # 1. FIXED PARAMS (from reports)
    beta = CB_BETA 
    gamma = FOCAL_GAMMA
    temperature = TEMP

    # 2. SUGGEST SCHEDULER PARAMS
    scheduler_type = trial.suggest_categorical("scheduler", ["linear", "cosine", "step", "plateau"])
    
    scheduler_params = {}
    if scheduler_type == "linear":
        scheduler_params["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.3)
    elif scheduler_type == "cosine":
        scheduler_params["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.3)
        scheduler_params["min_lr_ratio"] = trial.suggest_float("min_lr_ratio", 0.0, 0.2)
    elif scheduler_type == "step":
        scheduler_params["step_size"] = trial.suggest_int("step_size", 1, EPOCHS-1)
        scheduler_params["gamma"] = trial.suggest_float("gamma", 0.1, 0.9)
    elif scheduler_type == "plateau":
        scheduler_params["patience"] = trial.suggest_int("patience", 1, 3)
        scheduler_params["factor"] = trial.suggest_float("factor", 0.1, 0.9)

    print(f"\n🧪 TRIAL {trial.number} | Scheduler: {scheduler_type} | Params: {scheduler_params}")

    # 3. SETUP MODEL & OPTIMIZER
    # Loss Weights
    current_weights = get_weights_dynamic(global_counts, len(label2id), beta, DEVICE)
    loss_fn = FocalLoss(alpha=current_weights, gamma=gamma)
    
    # Model Reset
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
    
    optimizer = optim.AdamW([
        {"params": model_encoder.parameters(), "lr": FIXED_LR_MLP},
        {"params": proj.parameters(), "lr": FIXED_LR_PROJ}
    ], weight_decay=WEIGHT_DECAY)
    
    # 4. SETUP SCHEDULER
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = None
    step_scheduler_in_batch = False # True for linear/cosine, False for step/plateau
    
    if scheduler_type == "linear":
        num_warmup_steps = int(num_training_steps * scheduler_params["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        step_scheduler_in_batch = True
        
    elif scheduler_type == "cosine":
        num_warmup_steps = int(num_training_steps * scheduler_params["warmup_ratio"])
        # Handling min_lr via Transformers cosine isn't direct in get_cosine_schedule_with_warmup generally (it goes to 0), 
        # but torch.optim.lr_scheduler.CosineAnnealingLR supports eta_min. 
        # But let's stick to transformers one for consistency if possible, or use custom lambda. 
        # Simpler: use Transformers get_cosine_schedule_with_warmup (goes to 0) but maybe that's fine.
        # If user wants "min_lr_ratio", maybe I should use Torch's CosineAnnealingWarmRestarts or similar?
        # Let's stick to standard transformers cosine for now, but maybe ignore min_lr_ratio if library doesn't support it easily 
        # OR use CosineAnnealingLR from torch.
        # Let's use Torch CosineAnnealingLR for proper min_lr support. 
        # NOTE: Torch schedulers usually step per epoch, but CosineAnnealingLR can step per iteration if T_max is total steps.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=num_training_steps, 
                                                         eta_min=FIXED_LR_MLP * scheduler_params["min_lr_ratio"])
        step_scheduler_in_batch = True
        
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=scheduler_params["step_size"], 
                                              gamma=scheduler_params["gamma"])
        step_scheduler_in_batch = False # Step at epoch end
        
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         patience=scheduler_params["patience"], 
                                                         factor=scheduler_params["factor"])
        step_scheduler_in_batch = False # Step at epoch end with metric

    # 5. TRAINING LOOP
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            
            with torch.no_grad():
                # Text Encoder
                H_text = F.normalize(txt_enc(batch["input_ids"], batch["attention_mask"]).last_hidden_state, dim=-1)
            
            # Prompt Encoder
            desc_embeds = model_encoder(desc_input_ids, desc_attn_mask)
            pooled_mask = torch.ones(desc_embeds.shape[:2], device=DEVICE).long() if PROMPT_LEN else desc_attn_mask
            
            # Label Encoder
            lbl_out = lbl_enc(inputs_embeds=desc_embeds, attention_mask=pooled_mask).last_hidden_state
            mask_exp = pooled_mask.unsqueeze(-1).float()
            pooled = (lbl_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            H_label = F.normalize(proj(pooled), dim=-1)
            
            # Logits & Loss
            logits = torch.matmul(H_text, H_label.T) / temperature
            loss = loss_fn(logits.view(-1, len(label2id)), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 1.0)
            optimizer.step()
            
            if step_scheduler_in_batch and scheduler is not None:
                scheduler.step()
                
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Epoch Checkpoint
        if not step_scheduler_in_batch and scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
                
        trial.report(avg_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print(f"🏆 BEST PARAMS: {study.best_params}")
