# -*- coding: utf-8 -*-
"""
Training MLP Prompt Encoder on FULL Imbalanced Dataset.
Strategy: Inverse Frequency Class Weights + Unfrozen Projection + Differential Learning Rates.
"""

import json
import torch
import torch.nn.functional as F
import time
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
# --- SWITCH PRINCIPALE ---
TRAIN_PROJECTION = True
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10

# LEARNING RATES SEPARATI
LR_MLP = 5e-4       # Solitamente più alto (è un modulo nuovo)
LR_PROJ = 5e-4      # Solitamente più basso (è pre-addestrato, va solo raffinato)

WEIGHT_DECAY = 0.01
TEMPERATURE = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
RANDOM_SEED = 42
DROPOUT_RATE = 0.1
GAMMA_FOCAL_LOSS = 3.0

DATASET_PATH = "../dataset/dataset_tokenlevel_simple.json" 
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ CLASSE MLP PROMPT ENCODER
# ==========================================================
class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, hidden_dim=None, dropout=0.1):
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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.norm(x + self.mlp(x))

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
    dropout=DROPOUT_RATE
).to(DEVICE)

print(f"✨ MLP Prompt Encoder creato.")

# ==========================================================
# 3️⃣ DATASET & CALCOLO PESI
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

def get_weighted_loss_params(dataset_path, label2id, device):
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
    
    print("\n⚖️  CALCOLO PESI CLASSI (Inverse Frequency):")
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            w = total_tags / (num_classes * count)
        else:
            w = 1.0
        weights[label_id] = w
        print(f"   🔹 {label_name.ljust(15)}: {str(count).rjust(6)} occorrenze -> Peso Loss: {w:.4f}")
        
    return weights

# --- LOAD ---
ds = TokenJsonDataset(DATASET_PATH, txt_tok)
dataset_size = len(ds)
print(f"📊 Dimensione dataset: {dataset_size} esempi")

class_weights = get_weighted_loss_params(DATASET_PATH, label2id, DEVICE)
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
    print(f"🔗 Aggiunta Projection Layer all'optimizer con LR={LR_PROJ}")
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

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
            
        soft_embeds = prompt_encoder(desc_input_ids)
        outputs = lbl_enc(inputs_embeds=soft_embeds, attention_mask=desc_attn_mask)
        
        mask_exp = desc_attn_mask.unsqueeze(-1).float()
        pooled = torch.sum(outputs.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        label_matrix = F.normalize(proj(pooled), dim=-1)
        
        logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE
        loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP) # Clip gradients generically
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
                'train_projection': TRAIN_PROJECTION, 
                'dataset_size': dataset_size,
                'lr_mlp': LR_MLP,
                'lr_proj': LR_PROJ
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

if best_model_state is not None:
    os.makedirs("savings", exist_ok=True)
    proj_tag = "PROJ-TRUE" if TRAIN_PROJECTION else "PROJ-FALSE"
    # Nome file include entrambi i LR
    filename = f"mlp_FULL_{proj_tag}DATA{dataset_size}_lrMLP{LR_MLP}_lrPROJ{LR_PROJ}_ep{EPOCHS}_focal.pt"
    save_path = os.path.join("savings", filename)
    torch.save(best_model_state, save_path)
    print(f"💾 Modello salvato in {save_path}")
else:
    print("⚠️ Nessun modello salvato.")