#https://www.kaggle.com/code/alessandrobecci/promptencoderglinermonoencodertraining/
# -*- coding: utf-8 -*-
# SPAN-LEVEL LOSS VERSION (NO 'O' CLASS)
# Difference from train_mono.py: loss is computed on span-level representations
# instead of per-token. Each span's tokens are mean-pooled, then classified.
# The 'O' (background) class is completely excluded from training and evaluation.
import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import math
import time
import gc
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

# 🧹 PULIZIA MEMORIA INIZIALE
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
BATCH_SIZE = 8 
EPOCHS = 20

# LEARNING RATES
#LR_MLP = 3e-4
LR_MLP = 0.00017896280333089907
LR_BACKBONE = 0.0 # Frozen by default

WEIGHT_DECAY = 0.01
#0.05
TEMPERATURE = 0.14792401166908015
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
RANDOM_SEED = 42
DROPOUT_RATE = 0.1

PROMPT_LEN = 32
POOLING_MODE = "conv1d"

GAMMA_FOCAL_LOSS = 5.893415878096565
CB_BETA = 0.9999
WEIGHT_STRATEGY = "ClassBalanced"

EARLY_STOPPING_PATIENCE = 5

# Paths
if is_running_on_kaggle():
    input_dir = "/kaggle/input/datasets/alessandrobecci/jnlpa-18-5k15-3-5-complete/"
    TRAIN_PATH = input_dir + "dataset_span_mono.json"
    VAL_PATH = input_dir + "val_dataset_span_mono.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH = input_dir + "label2id.json"
    MODEL_NAME = '/kaggle/input/datasets/alessandrobecci/gliner2-1small/'
else:
    input_dir = "../" 
    TRAIN_PATH = input_dir + "dataset/dataset_span_mono.json"
    VAL_PATH = input_dir + "dataset/val_dataset_span_mono.json"
    LABEL2DESC_PATH = input_dir + "dataset/label2desc.json"
    LABEL2ID_PATH = input_dir + "dataset/label2id.json"
    MODEL_NAME = "urchade/gliner_small-v2.1"

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
# 3️⃣ PREPARAZIONE MODELLO
# ==========================================================
print("📦 Caricamento modello...")
model_wrapper = GLiNER.from_pretrained(MODEL_NAME)
model = model_wrapper.model
tokenizer = model_wrapper.data_processor.transformer_tokenizer

# Backbone Reference
backbone = model.token_rep_layer.bert_layer.model
original_word_embeddings = backbone.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

# 🔒 FREEZE BACKBONE
for p in backbone.parameters():
    p.requires_grad = False
print(f"✅ Backbone Configurato (Frozen). Dim: {embed_dim}")

# ==========================================================
# 4️⃣ DATASET & CALCOLO PESI
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id_full = json.load(f)

# 🚫 RIMUOVI CLASSE 'O' e RIMAPPA GLI ID
# Identifica l'ID originale della classe O
O_ORIGINAL_ID = label2id_full.get('O', None)
print(f"🚫 Classe 'O' trovata con ID originale: {O_ORIGINAL_ID} - verrà ESCLUSA dal training")

# Crea label2id senza O, con ID rimappati a 0..N-1
label2id = {}
new_id = 0
original_to_new = {}  # Mappa: original_id -> new_id (escluso O)
for name, orig_id in sorted(label2id_full.items(), key=lambda x: x[1]):
    if name == 'O':
        continue
    label2id[name] = new_id
    original_to_new[orig_id] = new_id
    new_id += 1

id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)

print(f"📋 Label rimappate (senza O): {label2id}")
print(f"📋 Mappa original->new: {original_to_new}")
print(f"📋 Num labels (senza O): {num_labels}")

# Calculate Max Text Length allowed
# Model Max Len (512) - (NumLabels * PromptLen) - SpecialTokens(3: CLS, SEP, SEP)
MAX_MODEL_LEN = 512
TOTAL_PROMPT_TOKENS = num_labels * PROMPT_LEN
MAX_TEXT_LEN = MAX_MODEL_LEN - TOTAL_PROMPT_TOKENS - 5 # Safety buffer
print(f"📏 Max Text Length imposto a: {MAX_TEXT_LEN} (Prompt Tokens: {TOTAL_PROMPT_TOKENS})")

# Tokenize Descriptions
desc_texts = [label2desc[name] for name in label_names]
desc_inputs = tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

# Create/Load Prompt Encoder
prompt_encoder = MLPPromptEncoder(
    original_word_embeddings, 
    vocab_size, 
    embed_dim, 
    dropout=DROPOUT_RATE,
    prompt_len=PROMPT_LEN,
    pooling_mode=POOLING_MODE,
    max_seq_len=desc_input_ids.shape[1]
).to(DEVICE)

print(f"✨ MLP Prompt Encoder creato. Prompt Length: {PROMPT_LEN}, Mode: {POOLING_MODE}")

# ==========================================================
# SPAN DATASET
# ==========================================================
class SpanJsonDataset(Dataset):
    """
    Span-level dataset. Each record has:
      - tokenized_text: list of word-level tokens
      - ner: list of [start, end, label_str] spans (inclusive indices)
    
    We tokenize words with the transformer tokenizer (word-level -> subword-level),
    then build a mapping from word indices to subword indices so that spans
    can be expressed in subword space.
    """
    def __init__(self, path_json, tokenizer, max_len=512):
        print(f"📖 Leggendo {path_json}...")
        with open(path_json, "r", encoding="utf-8") as f: 
            self.records = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        words = rec["tokenized_text"]
        spans = rec["ner"]  # [[start, end, label_str], ...]
        
        # Tokenize each word into subword tokens
        # Build word -> subword index mapping
        all_input_ids = []
        word_to_subword_start = []
        word_to_subword_end = []
        
        for word in words:
            subwords = self.tok.encode(word, add_special_tokens=False)
            word_to_subword_start.append(len(all_input_ids))
            all_input_ids.extend(subwords)
            word_to_subword_end.append(len(all_input_ids) - 1)
        
        # Truncate to max_len
        if len(all_input_ids) > self.max_len:
            all_input_ids = all_input_ids[:self.max_len]
        
        # Build span annotations in subword space (SKIP 'O' spans)
        span_labels = []  # [(sub_start, sub_end, new_label_id), ...]
        for span in spans:
            word_start, word_end, label_str = span[0], span[1], str(span[2])
            original_label_id = int(label_str)
            
            # Skip O class spans
            if original_label_id == O_ORIGINAL_ID:
                continue
            
            # Remap to new ID
            new_label_id = original_to_new[original_label_id]
            
            # Get subword indices for this span
            sub_start = word_to_subword_start[word_start]
            sub_end = word_to_subword_end[word_end]
            
            # Check if span is within truncated length
            if sub_start >= len(all_input_ids):
                continue
            sub_end = min(sub_end, len(all_input_ids) - 1)
            
            span_labels.append((sub_start, sub_end, new_label_id))
            
        return {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor([1] * len(all_input_ids)),
            "span_labels": span_labels,  # list of (start, end, label_id) tuples
        }

def collate_span_batch(batch, pad_id):
    """
    Collate for span dataset. Pads input_ids and attention_mask,
    keeps span_labels as a list of lists (variable length per sample).
    """
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    all_span_labels = []
    
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        all_span_labels.append(ex["span_labels"])
    
    return {"input_ids": input_ids, "attention_mask": attn_mask, "span_labels": all_span_labels}

def get_cb_weights_span(dataset_path, label2id, device, beta=0.9999):
    """
    Compute Class-Balanced weights from span dataset.
    Counts spans per label (not tokens).
    """
    with open(dataset_path, "r", encoding="utf-8") as f: 
        data = json.load(f)
    label_counts = Counter()
    for record in data:
        for span in record["ner"]:
            original_label_id = int(str(span[2]))
            # Skip O class
            if original_label_id == O_ORIGINAL_ID:
                continue
            new_label_id = original_to_new[original_label_id]
            label_counts[new_label_id] += 1
                
    num_classes = len(label2id)
    weights = torch.ones(num_classes).to(device)
    print(f"\n  CALCOLO PESI (Class Balanced - Beta {beta}) [SPAN-LEVEL, NO 'O']:")
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else: weights[label_id] = 0.0 
    weights = weights / weights.sum() * num_classes
    for label_name, label_id in label2id.items():
        print(f"   🔹 {label_name.ljust(15)}: {str(label_counts.get(label_id,0)).rjust(6)} -> Peso: {weights[label_id].item():.4f}")
    return weights

print("📊 Loading Dataset...")
train_ds = SpanJsonDataset(TRAIN_PATH, tokenizer, max_len=MAX_TEXT_LEN)
val_ds = SpanJsonDataset(VAL_PATH, tokenizer, max_len=MAX_TEXT_LEN)

print(f"🔪 Train size: {len(train_ds)} | Valid size: {len(val_ds)}")

class_weights = get_cb_weights_span(TRAIN_PATH, label2id, DEVICE, beta=CB_BETA)
ce_loss = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_span_batch(b, tokenizer.pad_token_id))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_span_batch(b, tokenizer.pad_token_id))

# ==========================================================
# 5️⃣ TRAINING LOOP
# ==========================================================
optimizer_grouped_parameters = [{"params": prompt_encoder.parameters(), "lr": LR_MLP}]
optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

backbone.to(DEVICE)
backbone.eval() # Backbone Frozen

best_loss = float('inf')
best_model_state = None
patience_counter = 0

# Timing tracking
epoch_times = []  # List of dicts: {'epoch': int, 'train_time': float, 'val_time': float, 'total_time': float}

print(f"\n🚀 Inizio Training (SPAN-LEVEL LOSS, NO 'O') | MLP LR: {LR_MLP}")

def compute_span_loss(text_reps, prompt_vectors, span_labels_batch, loss_fn, temperature, num_labels):
    """
    Compute span-level loss.
    
    For each sample in the batch:
      1. For each span (start, end, label), mean-pool the text_reps[start:end+1]
      2. Compute similarity between span representation and all prompt vectors
      3. Collect all span logits and labels, then compute loss
    
    Args:
        text_reps: (B, TextLen, D) - text representations from backbone
        prompt_vectors: (B, NumLabels, D) - label prompt representations 
        span_labels_batch: list of lists of (start, end, label_id) tuples
        loss_fn: loss function (FocalLoss)
        temperature: temperature for similarity scaling
        num_labels: number of label classes
    
    Returns:
        loss: scalar loss value
    """
    all_span_logits = []
    all_span_targets = []
    
    B = text_reps.shape[0]
    
    # Normalize once
    H_text = F.normalize(text_reps, dim=-1)
    H_prompts = F.normalize(prompt_vectors, dim=-1)
    
    for b in range(B):
        spans = span_labels_batch[b]
        if len(spans) == 0:
            continue
            
        for (s_start, s_end, label_id) in spans:
            # Mean-pool token representations within the span
            # span_reps shape: (span_len, D)
            span_reps = H_text[b, s_start:s_end+1, :]
            # Mean pool to get (D,)
            span_vec = span_reps.mean(dim=0)
            
            # Compute similarity with all label prompts: (NumLabels,)
            logits = torch.mv(H_prompts[b], span_vec) / temperature
            
            all_span_logits.append(logits)
            all_span_targets.append(label_id)
    
    if len(all_span_logits) == 0:
        return torch.tensor(0.0, device=text_reps.device, requires_grad=True)
    
    # Stack: (NumSpans, NumLabels) and (NumSpans,)
    all_span_logits = torch.stack(all_span_logits, dim=0)
    all_span_targets = torch.tensor(all_span_targets, device=text_reps.device, dtype=torch.long)
    
    return loss_fn(all_span_logits, all_span_targets)


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    # TRAIN
    train_start_time = time.time()
    prompt_encoder.train()
    total_train_loss = 0
    print(f"Epoch {epoch}/{EPOCHS} [Train] started...")
    
    for batch in train_loader:
        # Move only tensor fields to device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        span_labels_batch = batch["span_labels"]  # stays on CPU (list of lists)
        
        optimizer.zero_grad()
        
        # 1. Generate Soft Prompts
        # (NumLabels, PLen, D)
        soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask) 
        
        # Flatten Prompts to a single sequence per sample
        # (NumLabels * PLen, D)
        soft_prompts_flat = soft_prompts.view(-1, embed_dim) 
        prompts_len = soft_prompts_flat.shape[0]
        
        # Expand for Batch: (B, TotalPromptLen, D)
        batch_soft_prompts = soft_prompts_flat.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        
        # 2. Get Text Embeddings
        # (B, TextLen, D)
        text_embeds = backbone.embeddings(input_ids)
        
        # 3. Concatenate: [CLS] + Prompts + [SEP] + Text + [SEP]
        cls_token = torch.tensor([[tokenizer.cls_token_id]] * input_ids.shape[0], device=DEVICE)
        sep_token = torch.tensor([[tokenizer.sep_token_id]] * input_ids.shape[0], device=DEVICE)
        
        cls_embed = backbone.embeddings(cls_token) # (B, 1, D)
        sep_embed = backbone.embeddings(sep_token) # (B, 1, D)
        
        inputs_embeds = torch.cat([
            cls_embed,
            batch_soft_prompts,
            sep_embed,
            text_embeds,
            sep_embed
        ], dim=1)
        
        # ATTENTION MASK
        B = input_ids.shape[0]
        prompt_mask = torch.ones((B, prompts_len), device=DEVICE)
        cls_mask = torch.ones((B, 1), device=DEVICE)
        sep_mask = torch.ones((B, 1), device=DEVICE)
        
        full_mask = torch.cat([
            cls_mask,
            prompt_mask,
            sep_mask,
            attention_mask, # Text mask (with 0 for padding)
            sep_mask
        ], dim=1)
        
        # 4. Forward Backbone
        outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)) 
        sequence_output = outputs.last_hidden_state
        
        # 5. Extract Representations
        # Indices in sequence: CLS (1) + Prompts (PLen) + SEP (1) -> Start of Text is 1 + PLen + 1
        text_start = 1 + prompts_len + 1
        text_end = text_start + input_ids.shape[1]
        
        text_reps = sequence_output[:, text_start:text_end, :] # (B, TextLen, D)
        
        # Prompt Reps: 1 to 1 + PLen
        prompt_reps_seq = sequence_output[:, 1:1+prompts_len, :] # (B, NumLabels*PLen, D)
        
        # Reshape back to (B, NumLabels, PLen, D)
        prompt_reps_reshaped = prompt_reps_seq.view(B, soft_prompts.shape[0], soft_prompts.shape[1], embed_dim)
        
        # Pool Prompts to get 1 vector per label (Mean over Prompt Length)
        prompt_vectors = prompt_reps_reshaped.mean(dim=2) # (B, NumLabels, D)
        
        # 6. SPAN-LEVEL Similarity & Loss (MAIN DIFFERENCE FROM train_mono.py)
        loss = compute_span_loss(text_reps, prompt_vectors, span_labels_batch, ce_loss, TEMPERATURE, num_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    
    # VALIDATION
    val_start_time = time.time()
    prompt_encoder.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            span_labels_batch = batch["span_labels"]
            
            # Same construction as above
            soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            soft_prompts_flat = soft_prompts.view(-1, embed_dim)
            prompts_len = soft_prompts_flat.shape[0]
            batch_soft_prompts = soft_prompts_flat.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            
            text_embeds = backbone.embeddings(input_ids)
            cls_token = torch.tensor([[tokenizer.cls_token_id]] * input_ids.shape[0], device=DEVICE)
            sep_token = torch.tensor([[tokenizer.sep_token_id]] * input_ids.shape[0], device=DEVICE)
            cls_embed = backbone.embeddings(cls_token)
            sep_embed = backbone.embeddings(sep_token)
            
            inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
            
            B = input_ids.shape[0]
            full_mask = torch.cat([
                torch.ones((B, 1), device=DEVICE),
                torch.ones((B, prompts_len), device=DEVICE),
                torch.ones((B, 1), device=DEVICE),
                attention_mask,
                torch.ones((B, 1), device=DEVICE)
            ], dim=1)
            
            outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2))
            sequence_output = outputs.last_hidden_state
            
            text_start = 1 + prompts_len + 1
            text_end = text_start + input_ids.shape[1]
            text_reps = sequence_output[:, text_start:text_end, :]
            
            prompt_reps_seq = sequence_output[:, 1:1+prompts_len, :]
            prompt_reps_reshaped = prompt_reps_seq.view(B, soft_prompts.shape[0], soft_prompts.shape[1], embed_dim)
            prompt_vectors = prompt_reps_reshaped.mean(dim=2)
            
            # SPAN-LEVEL loss
            loss = compute_span_loss(text_reps, prompt_vectors, span_labels_batch, ce_loss, TEMPERATURE, num_labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    
    epoch_end_time = time.time()
    total_epoch_time = epoch_end_time - epoch_start_time
    
    # Store timing information
    epoch_times.append({
        'epoch': epoch,
        'train_time': train_time,
        'val_time': val_time,
        'total_time': total_epoch_time
    })
    
    # Print epoch summary with detailed timing
    train_mins, train_secs = divmod(train_time, 60)
    val_mins, val_secs = divmod(val_time, 60)
    total_mins, total_secs = divmod(total_epoch_time, 60)
    
    print(f"\n{'='*80}")
    print(f"Epoch {epoch}/{EPOCHS} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Training Time:   {int(train_mins):2d}m {int(train_secs):02d}s")
    print(f"  Validation Time: {int(val_mins):2d}m {int(val_secs):02d}s")
    print(f"  Total Time:      {int(total_mins):2d}m {int(total_secs):02d}s")
    print(f"{'='*80}\n")
    
    # Save & Early Stopping (Structure Updated)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        
        # Save structured state like reference script
        best_model_state = {
            'prompt_encoder': prompt_encoder.state_dict(),
            'config': {
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'input_dir': input_dir,
                'dataset_size_train': len(train_ds),
                'dataset_size_val': len(val_ds),
                'train_size': len(train_ds),
                'val_size': len(val_ds),
                'random_seed': RANDOM_SEED,
                'lr_mlp': LR_MLP,
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP,
                'warmup_ratio': WARMUP_RATIO,
                'temperature': TEMPERATURE,
                'dropout_rate': DROPOUT_RATE,
                'weight_strategy': WEIGHT_STRATEGY,
                'gamma_focal_loss': GAMMA_FOCAL_LOSS,
                'cb_beta': CB_BETA,
                'validation_split': False,
                'prompt_len': PROMPT_LEN,
                'pooling_mode': POOLING_MODE,
                'max_text_len': MAX_TEXT_LEN,
                'patience': EARLY_STOPPING_PATIENCE,
                'loss_level': 'span',  # NEW: indicates span-level loss
                'exclude_O': True,  # O class excluded from training
            }
        }
        
        # Move state to CPU to save memory/compatibility
        for key in best_model_state:
            if isinstance(best_model_state[key], dict):
                best_model_state[key] = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in best_model_state[key].items()}

        print(f"  → Nuovo best model (Val Loss: {best_loss:.4f}) - Stato aggiornato in memoria")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ⏳ Nessun miglioramento. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early Stopping attivo! Nessun miglioramento per {EARLY_STOPPING_PATIENCE} epoche.")
            break

# Print timing summary table
print(f"\n{'='*80}")
print(f"⏱️  TIMING SUMMARY")
print(f"{'='*80}")
print(f"{'Epoch':<8} {'Train Time':<15} {'Val Time':<15} {'Total Time':<15}")
print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")

for timing in epoch_times:
    train_mins, train_secs = divmod(timing['train_time'], 60)
    val_mins, val_secs = divmod(timing['val_time'], 60)
    total_mins, total_secs = divmod(timing['total_time'], 60)
    
    print(f"{timing['epoch']:<8} "
          f"{int(train_mins):2d}m {int(train_secs):02d}s{' '*7} "
          f"{int(val_mins):2d}m {int(val_secs):02d}s{' '*7} "
          f"{int(total_mins):2d}m {int(total_secs):02d}s")

# Calculate and print averages
if epoch_times:
    avg_train_time = sum(t['train_time'] for t in epoch_times) / len(epoch_times)
    avg_val_time = sum(t['val_time'] for t in epoch_times) / len(epoch_times)
    avg_total_time = sum(t['total_time'] for t in epoch_times) / len(epoch_times)
    total_training_time = sum(t['total_time'] for t in epoch_times)
    
    avg_train_mins, avg_train_secs = divmod(avg_train_time, 60)
    avg_val_mins, avg_val_secs = divmod(avg_val_time, 60)
    avg_total_mins, avg_total_secs = divmod(avg_total_time, 60)
    total_mins, total_secs = divmod(total_training_time, 60)
    
    print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'Average':<8} "
          f"{int(avg_train_mins):2d}m {int(avg_train_secs):02d}s{' '*7} "
          f"{int(avg_val_mins):2d}m {int(avg_val_secs):02d}s{' '*7} "
          f"{int(avg_total_mins):2d}m {int(avg_total_secs):02d}s")
    print(f"\n{'Total Training Time:':<25} {int(total_mins):3d}m {int(total_secs):02d}s")

print(f"{'='*80}")
print(f"\n✅ Training completato (SPAN-LEVEL, NO 'O'). Best Validation Loss: {best_loss:.4f}")

# ==========================================================
# 6️⃣ SALVATAGGIO FINALE
# ==========================================================
from datetime import datetime, timedelta

if best_model_state is not None:
    os.makedirs("savings", exist_ok=True)
    
    now = datetime.now()
    if is_running_on_kaggle(): 
        now = now + timedelta(hours=1)
    
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"mlp_mono_focal_cbclass_span_noO_val-{timestamp}.pt"
    save_path = os.path.join("savings", filename)
    
    torch.save(best_model_state, save_path)
    print(f"💾 Modello salvato in {save_path}")
else:
    print("⚠️ Nessun modello salvato (Loss mai migliorata?)")

# ==========================================================
# 7️⃣ TEST AUTOMATICO SUL MODELLO MIGLIORE
# ==========================================================
if best_model_state is not None:
    print(f"\n{'='*80}")
    print(f"🧪 AVVIO TEST AUTOMATICO SUL MODELLO MIGLIORE (SPAN-LEVEL, NO 'O')")
    print(f"{'='*80}\n")
    
    # Determine test path
    if is_running_on_kaggle():
        TEST_PATH = input_dir + "test_dataset_span_mono.json"
    else:
        TEST_PATH = input_dir + "dataset/test_dataset_span_mono.json"
    
    # Check if test file exists
    if not os.path.exists(TEST_PATH):
        print(f"⚠️ File di test non trovato: {TEST_PATH}")
        print("   Test automatico saltato.")
    else:
        # Import metrics
        from sklearn.metrics import precision_recall_fscore_support, classification_report
        
        # Load best model state into prompt_encoder
        prompt_encoder.load_state_dict(best_model_state['prompt_encoder'])
        prompt_encoder.eval()
        print(f"✅ Modello migliore caricato (Val Loss: {best_loss:.4f})")
        
        # Load test data
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"📊 Caricati {len(test_data)} record di test\n")
        
        # All labels are relevant (O was already excluded)
        relevant_label_ids = list(range(num_labels))
        relevant_label_names = label_names
        print(f"ℹ️  Tutte le {num_labels} classi sono rilevanti (O già esclusa).\n")
        
        # Pre-calculate prompt embeddings
        with torch.no_grad():
            soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            soft_prompts_flat = soft_prompts.view(-1, embed_dim)
            prompts_len_total = soft_prompts_flat.shape[0]
        
        # Run inference - SPAN LEVEL
        y_true = []
        y_pred = []
        
        print("🔍 Esecuzione inferenza sul test set (SPAN-LEVEL, NO 'O')...")
        with torch.no_grad():
            for rec in tqdm(test_data, desc="Testing"):
                words = rec["tokenized_text"]
                spans = rec["ner"]
                
                # Tokenize each word into subword tokens
                all_inp_ids = []
                word_to_subword_start = []
                word_to_subword_end = []
                
                for word in words:
                    subwords = tokenizer.encode(word, add_special_tokens=False)
                    word_to_subword_start.append(len(all_inp_ids))
                    all_inp_ids.extend(subwords)
                    word_to_subword_end.append(len(all_inp_ids) - 1)
                
                # Truncate to max text length
                if len(all_inp_ids) > MAX_TEXT_LEN:
                    all_inp_ids = all_inp_ids[:MAX_TEXT_LEN]
                
                if len(all_inp_ids) == 0:
                    continue
                
                input_tensor = torch.tensor([all_inp_ids], device=DEVICE)
                attn_mask = torch.ones_like(input_tensor)
                
                # 1. Expand prompts for batch (batch=1)
                batch_soft_prompts = soft_prompts_flat.unsqueeze(0)
                
                # 2. Text embeddings
                text_embeds = backbone.embeddings(input_tensor)
                
                # 3. Concatenate: [CLS] Prompts [SEP] Text [SEP]
                cls_token = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
                sep_token = torch.tensor([[tokenizer.sep_token_id]], device=DEVICE)
                cls_embed = backbone.embeddings(cls_token)
                sep_embed = backbone.embeddings(sep_token)
                
                inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
                
                # Build attention mask
                B = 1
                full_mask = torch.cat([
                    torch.ones((B, 1), device=DEVICE),           # CLS
                    torch.ones((B, prompts_len_total), device=DEVICE),  # Prompts
                    torch.ones((B, 1), device=DEVICE),           # SEP
                    attn_mask,                                    # Text
                    torch.ones((B, 1), device=DEVICE)            # SEP
                ], dim=1)
                
                # 4. Forward through encoder
                outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2))
                sequence_output = outputs.last_hidden_state
                
                # 5. Extract text representations
                text_start = 1 + prompts_len_total + 1
                text_end = text_start + len(all_inp_ids)
                text_reps = sequence_output[:, text_start:text_end, :]
                
                # 6. Extract and pool prompt representations
                prompt_reps_seq = sequence_output[:, 1:1+prompts_len_total, :]
                prompt_reps_reshaped = prompt_reps_seq.view(1, num_labels, PROMPT_LEN, embed_dim)
                prompt_vectors = prompt_reps_reshaped.mean(dim=2)
                
                # 7. Normalize
                H_text = F.normalize(text_reps, dim=-1)
                H_prompts = F.normalize(prompt_vectors, dim=-1)
                
                # 8. For each span, compute prediction (SKIP O spans)
                for span in spans:
                    word_start, word_end, label_str = span[0], span[1], str(span[2])
                    original_label = int(label_str)
                    
                    # Skip O class spans
                    if original_label == O_ORIGINAL_ID:
                        continue
                    
                    true_label = original_to_new[original_label]
                    
                    sub_start = word_to_subword_start[word_start]
                    sub_end = word_to_subword_end[word_end]
                    
                    # Check if span is within truncated length
                    if sub_start >= len(all_inp_ids):
                        continue
                    sub_end = min(sub_end, len(all_inp_ids) - 1)
                    
                    # Mean-pool span tokens
                    span_reps = H_text[0, sub_start:sub_end+1, :]
                    span_vec = span_reps.mean(dim=0)
                    
                    # Similarity with all label prompts
                    logits = torch.mv(H_prompts[0], span_vec)
                    pred_label = logits.argmax().item()
                    
                    y_true.append(true_label)
                    y_pred.append(pred_label)
        
        # Calculate metrics
        print(f"\n{'='*80}")
        print(f"📊 RISULTATI TEST (SPAN-LEVEL, NO 'O')")
        print(f"{'='*80}\n")
        
        if len(y_true) == 0:
            print("⚠️ Nessun span valido trovato nel test set!")
        else:
            # Macro metrics
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
            )
            # Micro metrics
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
            )
            
            print(f"🏆 METRICHE GLOBALI (O esclusa dal training) [SPAN-LEVEL]:")
            print(f"   • MACRO - Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
            print(f"   • MICRO - Precision: {micro_p:.4f} | Recall: {micro_r:.4f} | F1: {micro_f1:.4f}")
            print(f"   • Spans valutati: {len(y_true):,}\n")
            
            # Detailed classification report
            class_report = classification_report(
                y_true, y_pred, 
                target_names=relevant_label_names, 
                labels=relevant_label_ids, 
                zero_division=0
            )
            print("📋 CLASSIFICATION REPORT:\n")
            print(class_report)
            
            print(f"{'='*80}\n")
