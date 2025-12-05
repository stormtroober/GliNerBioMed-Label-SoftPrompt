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

# ==========================================
# 🔧 PARAMETRI FISSI (Dalle ricerche precedenti)
# ==========================================
PROMPT_LEN = 32         # <--- INSERISCI VINCITORE
POOLING_MODE = "conv1d" # <--- INSERISCI VINCITORE
FIXED_LR_MLP = 1e-3     # <--- INSERISCI RISULTATO FILE 1
FIXED_LR_PROJ = 1e-3    # <--- INSERISCI RISULTATO FILE 1

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

# Pre-calcoliamo i count una volta sola per velocità
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
# 🎯 OPTUNA OBJECTIVE: LOSS & TEMP
# ==========================================
def objective(trial):
    # 1. SUGGEST PARAMETERS
    
    # Class Balanced Beta: valori tipici [0.9, 0.99, 0.999, 0.9999]
    # Usiamo una selezione categorica perché beta è molto sensibile
    beta_str = trial.suggest_categorical("cb_beta", ["0.9", "0.99", "0.999", "0.9999"])
    beta = float(beta_str)
    
    # Focal Loss Gamma: [0.0 = CrossEntropy standard, 5.0 = Focus estremo su hard examples]
    gamma = trial.suggest_float("focal_gamma", 3.5, 5.0, step=0.5)
    
    # Temperature: [0.01 molto "sharp", 0.1 più "smooth"]
    temperature = trial.suggest_float("temperature", 0.01, 0.15)

    print(f"\n🧪 TRIAL {trial.number} | Beta: {beta} | Gamma: {gamma} | Temp: {temperature:.3f}")

    # 2. CALCOLO PESI DINAMICO
    # I pesi cambiano in base al Beta scelto nel trial
    current_weights = get_weights_dynamic(global_counts, len(label2id), beta, DEVICE)
    loss_fn = FocalLoss(alpha=current_weights, gamma=gamma)

    # 3. SETUP MODEL & OPTIMIZER (Usa gli LR fissi)
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
    ], weight_decay=0.01)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(len(train_loader)*EPOCHS*0.1), 
                                                num_training_steps=len(train_loader)*EPOCHS)

    # 4. TRAINING LOOP
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            
            with torch.no_grad():
                H_text = F.normalize(txt_enc(batch["input_ids"], batch["attention_mask"]).last_hidden_state, dim=-1)
            
            # Encoder
            desc_embeds = model_encoder(desc_input_ids, desc_attn_mask)
            pooled_mask = torch.ones(desc_embeds.shape[:2], device=DEVICE).long() if PROMPT_LEN else desc_attn_mask
            
            lbl_out = lbl_enc(inputs_embeds=desc_embeds, attention_mask=pooled_mask).last_hidden_state
            mask_exp = pooled_mask.unsqueeze(-1).float()
            pooled = (lbl_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            H_label = F.normalize(proj(pooled), dim=-1)
            
            # QUI USIAMO LA TEMPERATURA DEL TRIAL
            logits = torch.matmul(H_text, H_label.T) / temperature
            
            loss = loss_fn(logits.view(-1, len(label2id)), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        trial.report(avg_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print(f"🏆 BEST PARAMS: {study.best_params}")
# 🧪 TRIAL 0 | Beta: 0.99 | Gamma: 5.0 | Temp: 0.079
# [I 2025-12-04 12:01:06,223] Trial 0 finished with value: 0.07043681477632703 and parameters: {'cb_beta': '0.99', 'focal_gamma': 5.0, 'temperature': 0.07856556223683704}. Best is trial 0 with value: 0.07043681477632703.

# 🧪 TRIAL 1 | Beta: 0.9999 | Gamma: 4.0 | Temp: 0.088
# [I 2025-12-04 12:02:09,563] Trial 1 finished with value: 0.06231805833080147 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.0, 'temperature': 0.08786710672514386}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 2 | Beta: 0.9 | Gamma: 3.5 | Temp: 0.144
# [I 2025-12-04 12:03:12,753] Trial 2 finished with value: 0.1532605242314218 and parameters: {'cb_beta': '0.9', 'focal_gamma': 3.5, 'temperature': 0.1438606076102245}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 3 | Beta: 0.9 | Gamma: 4.0 | Temp: 0.078
# [I 2025-12-04 12:04:15,913] Trial 3 finished with value: 0.0952897264704674 and parameters: {'cb_beta': '0.9', 'focal_gamma': 4.0, 'temperature': 0.07812430930879452}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 4 | Beta: 0.9 | Gamma: 4.0 | Temp: 0.074
# [I 2025-12-04 12:05:19,215] Trial 4 finished with value: 0.09541733619533008 and parameters: {'cb_beta': '0.9', 'focal_gamma': 4.0, 'temperature': 0.07442774600387471}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 5 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.105
# [I 2025-12-04 12:06:22,668] Trial 5 finished with value: 0.0691346173422246 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.10482822676080684}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 6 | Beta: 0.999 | Gamma: 4.0 | Temp: 0.111
# [I 2025-12-04 12:06:33,486] Trial 6 pruned. 

# 🧪 TRIAL 7 | Beta: 0.9 | Gamma: 4.5 | Temp: 0.047
# [I 2025-12-04 12:07:37,587] Trial 7 finished with value: 0.06734734052155592 and parameters: {'cb_beta': '0.9', 'focal_gamma': 4.5, 'temperature': 0.04685052911229522}. Best is trial 1 with value: 0.06231805833080147.

# 🧪 TRIAL 8 | Beta: 0.9 | Gamma: 4.5 | Temp: 0.132
# [I 2025-12-04 12:07:48,102] Trial 8 pruned. 

# 🧪 TRIAL 9 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.023
# [I 2025-12-04 12:08:51,871] Trial 9 finished with value: 0.032737455672667 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.5, 'temperature': 0.022544333482883407}. Best is trial 9 with value: 0.032737455672667.

# 🧪 TRIAL 10 | Beta: 0.9999 | Gamma: 3.5 | Temp: 0.011
# [I 2025-12-04 12:09:02,518] Trial 10 pruned. 

# 🧪 TRIAL 11 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.042
# [I 2025-12-04 12:10:05,279] Trial 11 finished with value: 0.03787350216055218 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.5, 'temperature': 0.042203718068175015}. Best is trial 9 with value: 0.032737455672667.

# 🧪 TRIAL 12 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.024
# [I 2025-12-04 12:11:09,093] Trial 12 finished with value: 0.0326535582919664 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.5, 'temperature': 0.024139518430248082}. Best is trial 12 with value: 0.0326535582919664.

# 🧪 TRIAL 13 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.013
# [I 2025-12-04 12:12:12,528] Trial 13 finished with value: 0.027257606670071807 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.013119440021081487}. Best is trial 13 with value: 0.027257606670071807.

# 🧪 TRIAL 14 | Beta: 0.99 | Gamma: 5.0 | Temp: 0.040
# [I 2025-12-04 12:12:23,309] Trial 14 pruned. 

# 🧪 TRIAL 15 | Beta: 0.999 | Gamma: 5.0 | Temp: 0.023
# [I 2025-12-04 12:12:34,056] Trial 15 pruned. 

# 🧪 TRIAL 16 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.059
# [I 2025-12-04 12:13:37,027] Trial 16 finished with value: 0.03762096218481849 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.05850769929829392}. Best is trial 13 with value: 0.027257606670071807.

# 🧪 TRIAL 17 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.014
# [I 2025-12-04 12:14:39,805] Trial 17 finished with value: 0.031112853225462044 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.5, 'temperature': 0.014344536038260304}. Best is trial 13 with value: 0.027257606670071807.

# 🧪 TRIAL 18 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.012
# [I 2025-12-04 12:15:43,236] Trial 18 finished with value: 0.026334033147255076 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.011641058260782156}. Best is trial 18 with value: 0.026334033147255076.

# 🧪 TRIAL 19 | Beta: 0.99 | Gamma: 5.0 | Temp: 0.059
# [I 2025-12-04 12:15:54,070] Trial 19 pruned. 

# 🧪 TRIAL 20 | Beta: 0.999 | Gamma: 5.0 | Temp: 0.033
# [I 2025-12-04 12:16:04,701] Trial 20 pruned. 

# 🧪 TRIAL 21 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.013
# [I 2025-12-04 12:16:15,350] Trial 21 pruned. 

# 🧪 TRIAL 22 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.010
# [I 2025-12-04 12:16:25,941] Trial 22 pruned. 

# 🧪 TRIAL 23 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.029
# [I 2025-12-04 12:17:29,032] Trial 23 finished with value: 0.033680394504077824 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 4.5, 'temperature': 0.029364482540967726}. Best is trial 18 with value: 0.026334033147255076.

# 🧪 TRIAL 24 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.057
# [I 2025-12-04 12:18:32,059] Trial 24 finished with value: 0.03720436271138584 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.05724880758215375}. Best is trial 18 with value: 0.026334033147255076.

# 🧪 TRIAL 25 | Beta: 0.9999 | Gamma: 4.5 | Temp: 0.049
# [I 2025-12-04 12:18:53,418] Trial 25 pruned. 

# 🧪 TRIAL 26 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.019
# [I 2025-12-04 12:19:56,759] Trial 26 finished with value: 0.027465461482164225 and parameters: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.018557537702155236}. Best is trial 18 with value: 0.026334033147255076.

# 🧪 TRIAL 27 | Beta: 0.9999 | Gamma: 5.0 | Temp: 0.032
# [I 2025-12-04 12:20:07,494] Trial 27 pruned. 

# 🧪 TRIAL 28 | Beta: 0.99 | Gamma: 5.0 | Temp: 0.038
# [I 2025-12-04 12:20:18,124] Trial 28 pruned. 

# 🧪 TRIAL 29 | Beta: 0.999 | Gamma: 5.0 | Temp: 0.022
# [I 2025-12-04 12:20:28,840] Trial 29 pruned. 
# 🏆 BEST PARAMS: {'cb_beta': '0.9999', 'focal_gamma': 5.0, 'temperature': 0.011641058260782156}

#TEST CON DATASET 15K
