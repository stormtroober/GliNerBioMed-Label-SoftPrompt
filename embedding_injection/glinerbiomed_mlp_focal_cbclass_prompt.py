# -*- coding: utf-8 -*-

"""
5000 for now is the best dataset size. But probably is just a matter of parameter tuning.
"""

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

def is_running_on_kaggle():
    # Kaggle monta i dataset in questa directory
    return os.path.exists('/kaggle/input')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
TRAIN_PROJECTION = True
BATCH_SIZE = 64
EPOCHS = 10

# LEARNING RATES SEPARATI
LR_MLP = 1e-3
LR_PROJ = 1e-3

WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
GRAD_CLIP = 1.0
WARMUP_PERCENTAGE = 0.15
RANDOM_SEED = 42
DROPOUT_RATE = 0.1

PROMPT_LEN = 16
POOLING_MODE = "adaptive_avg"  # "adaptive_avg", "attention"

GAMMA_FOCAL_LOSS = 4.5
CB_BETA = 0.9999
WEIGHT_STRATEGY = "ClassBalanced"

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
# 1️⃣ CLASSE MLP PROMPT ENCODER (CON POOLING)
# ==========================================================
class PromptPooler(nn.Module):
    """Riduce la sequenza da (B, seq_len, dim) a (B, prompt_len, dim)"""
    
    def __init__(self, embed_dim, prompt_len, mode="adaptive_avg"):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        
        if mode == "adaptive_avg":
            self.pooler = nn.AdaptiveAvgPool1d(prompt_len)
        elif mode == "adaptive_max":
            self.pooler = nn.AdaptiveMaxPool1d(prompt_len)
        elif mode == "attention":
            # Learnable query tokens che estraggono PROMPT_LEN rappresentazioni
            self.queries = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, seq_len, dim)
            attention_mask: (B, seq_len) - 1 per token validi, 0 per padding
        Returns:
            (B, prompt_len, dim)
        """
        B, seq_len, dim = x.shape
        
        if self.mode in ["adaptive_avg", "adaptive_max"]:
            # AdaptivePool lavora su ultima dim, quindi permute
            # (B, seq_len, dim) -> (B, dim, seq_len)
            x_t = x.transpose(1, 2)
            
            # Applica mask se presente (sostituisci padding con 0 per avg, -inf per max)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).float()  # (B, 1, seq_len)
                if self.mode == "adaptive_avg":
                    x_t = x_t * mask_expanded
                else:  # adaptive_max
                    x_t = x_t.masked_fill(mask_expanded == 0, float('-inf'))
            
            # (B, dim, seq_len) -> (B, dim, prompt_len)
            pooled = self.pooler(x_t)
            # (B, dim, prompt_len) -> (B, prompt_len, dim)
            return pooled.transpose(1, 2)
        
        elif self.mode == "attention":
            # Queries apprese: (1, prompt_len, dim) -> (B, prompt_len, dim)
            queries = self.queries.expand(B, -1, -1)
            
            # Key padding mask per attention (True = ignore)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
            
            # Cross-attention: queries attendono a x
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)  # Residual


class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg"):
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
        
        # Pooler opzionale
        self.pooler = None
        if prompt_len is not None:
            self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        
        # Applica pooling se configurato
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
# 3️⃣ PREPARAZIONE MODELLO
# ==========================================================
print("📦 Caricamento modello...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# 🔒 FREEZING
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = False

# Gestione dinamica della Projection
for p in proj.parameters():    
    p.requires_grad = TRAIN_PROJECTION

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

status_proj = "SBLOCCATA (Trainable)" if TRAIN_PROJECTION else "CONGELATA (Frozen)"
print(f"✅ Backbone Configurato. Projection: {status_proj}")

# Creazione Prompt Encoder
original_word_embeddings = lbl_enc.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

prompt_encoder = MLPPromptEncoder(
    original_word_embeddings, 
    vocab_size, 
    embed_dim, 
    dropout=DROPOUT_RATE,
    prompt_len=PROMPT_LEN,
    pooling_mode=POOLING_MODE
).to(DEVICE)

print(f"✨ MLP Prompt Encoder creato. Prompt Length: {PROMPT_LEN if PROMPT_LEN else 'Originale'}, Mode: {POOLING_MODE}")

# ==========================================================
# 3️⃣ DATASET & CALCOLO PESI (CLASS BALANCED)
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

# --- CLASS BALANCED WEIGHTS ---
def get_cb_weights(dataset_path, label2id, device, beta=0.9999):
    """
    Calcola i pesi usando il metodo 'Class Balanced Loss' (Cui et al., CVPR 2019).
    Formula: Weight = (1 - beta) / (1 - beta^N)
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    label_counts = Counter()
    total_tags = 0
    
    for record in data:
        for label_id in record["labels"]:
            if label_id != -100:
                label_counts[label_id] += 1
                total_tags += 1
                
    num_classes = len(label2id)
    weights = torch.ones(num_classes).to(device)
    
    print(f"\n  CALCOLO PESI (Class Balanced - Beta {beta}):")
    
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            # Calcolo Effective Number: (1 - beta^n) / (1 - beta)
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else:
            # Fallback per classi non presenti nel train (raro)
            weights[label_id] = 0.0 
        
    # Normalizzazione Cruciale: 
    # La somma dei pesi deve essere uguale al numero di classi.
    # Questo previene che la loss diventi gigante o minuscola globalmente.
    weights = weights / weights.sum() * num_classes
    
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        print(f"   🔹 {label_name.ljust(15)}: {str(count).rjust(6)} occorrenze -> Peso CB: {weights[label_id].item():.4f}")
        
    return weights

# --- LOAD ---
ds = TokenJsonDataset(DATASET_PATH, txt_tok)
dataset_size = len(ds)
print(f"📊 Dimensione dataset: {dataset_size} esempi")
import math
WARMUP_STEPS = round(math.ceil(dataset_size / BATCH_SIZE) * EPOCHS * WARMUP_PERCENTAGE)

class_weights = get_cb_weights(DATASET_PATH, label2id, DEVICE, beta=CB_BETA)

ce_loss = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id))

# ==========================================================
# 4️⃣ TRAINING LOOP (DIFFERENTIAL LEARNING RATES)
# ==========================================================

# 1. Gruppo parametri MLP (LR specifico)
optimizer_grouped_parameters = [
    {
        "params": prompt_encoder.parameters(),
        "lr": LR_MLP,
    }
]

# 2. Gruppo parametri Projection (LR specifico)
if TRAIN_PROJECTION:
    print(f" Aggiunta Projection Layer all'optimizer con LR={LR_PROJ}")
    optimizer_grouped_parameters.append({
        "params": proj.parameters(),
        "lr": LR_PROJ,
    })
else:
    print("🔒 Projection Layer esclusa dall'optimizer (Frozen)")

# Creazione Optimizer unico con gruppi separati
optimizer = optim.AdamW(
    optimizer_grouped_parameters, 
    weight_decay=WEIGHT_DECAY
)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader)*EPOCHS)

txt_enc.eval() 
lbl_enc.eval()
prompt_encoder.train()

if TRAIN_PROJECTION: proj.train()
else: proj.eval()

best_loss = float('inf')
best_model_state = None

print(f"\n🚀 Inizio Training | MLP LR: {LR_MLP} | PROJ LR: {LR_PROJ if TRAIN_PROJECTION else 'N/A'}")
print(f"🎯 Configurazione Loss: Class Balanced (Beta={CB_BETA}) + Focal (Gamma={GAMMA_FOCAL_LOSS})")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
            
        # MODIFICATO: passa attention_mask al prompt_encoder
        soft_embeds = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
        
        # Crea nuova attention mask per il label encoder (se pooling attivo)
        if PROMPT_LEN is not None:
            # Dopo pooling, tutti i token sono validi
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
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = {
            'prompt_encoder': prompt_encoder.state_dict(),
            'config': {
                # Parametri generali di training
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'dataset_size': dataset_size,
                'random_seed': RANDOM_SEED,
                
                # Learning Rates e Ottimizzazione
                'lr_mlp': LR_MLP,
                'lr_proj': LR_PROJ,
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP,
                'warmup_steps': WARMUP_STEPS,
                'warmup_percentage': WARMUP_PERCENTAGE,
                
                # Parametri Modello / Loss
                'temperature': TEMPERATURE,
                'dropout_rate': DROPOUT_RATE,
                'weight_strategy': WEIGHT_STRATEGY,
                'gamma_focal_loss': GAMMA_FOCAL_LOSS,
                'cb_beta': CB_BETA,

                #Parametri Prompt Pooling
                'prompt_len': PROMPT_LEN,
                'pooling_mode': POOLING_MODE,
            }
        }
        if TRAIN_PROJECTION:
            best_model_state['projection'] = proj.state_dict()
        
        # Spostamento su CPU
        for key in best_model_state:
            if isinstance(best_model_state[key], dict):
                best_model_state[key] = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in best_model_state[key].items()}
        
        print(f"  → Nuovo best loss: {best_loss:.4f}")

print(f"\n✅ Training completato. Best Loss: {best_loss:.4f}")

try:
    import google.colab # type: ignore
    IN_COLAB = True
except:
    IN_COLAB = False


from datetime import datetime, timedelta

if best_model_state is not None:
        os.makedirs("savings", exist_ok=True)
        
        now = datetime.now()
        
        # Se sei in Colab o Kaggle, aggiungi un'ora
        if IN_COLAB or is_running_on_kaggle():
            now = now + timedelta(hours=1)
        
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        filename = f"mlp_focal_cbclass-{timestamp}.pt"
        
        save_path = os.path.join("savings", filename)
        
        torch.save(best_model_state, save_path)
        print(f"💾 Modello salvato in {save_path}")
        
else:
        print("⚠️ Nessun modello salvato.")