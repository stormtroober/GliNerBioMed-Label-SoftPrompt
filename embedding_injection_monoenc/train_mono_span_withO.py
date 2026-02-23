#https://www.kaggle.com/code/alessandrobecci/promptencoderglinermonoencodertraining/
# -*- coding: utf-8 -*-
# SPAN-LEVEL LOSS VERSION — CON CLASSE 'O'
# Differenza da train_mono_span.py: la classe O è INCLUSA nel training.
# Il modello impara a predire O per gli span di background, e le 5 classi reali
# per gli span di entità. Questo è necessario per un estrattore NER reale.

import json
import torch
import torch.nn.functional as F
import os
import numpy as np
import math
import time
import gc
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from transformers import get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
BATCH_SIZE   = 8
EPOCHS       = 20
LR_MLP       = 0.00017896280333089907
LR_BACKBONE  = 0.0
WEIGHT_DECAY = 0.01
TEMPERATURE  = 0.14792401166908015
GRAD_CLIP    = 1.0
WARMUP_RATIO = 0.1
RANDOM_SEED  = 42
DROPOUT_RATE = 0.1
PROMPT_LEN   = 32
POOLING_MODE = "conv1d"
GAMMA_FOCAL_LOSS  = 5.893415878096565
CB_BETA           = 0.9999
WEIGHT_STRATEGY   = "ClassBalanced"
EARLY_STOPPING_PATIENCE = 5

# Parametri per il test estrattore
EXTRACTOR_THRESHOLD = 0.5
EXTRACTOR_MAX_SPAN  = 12

if is_running_on_kaggle():
    input_dir      = "/kaggle/input/datasets/alessandrobecci/jnlpa-18-5k15-3-5-complete/"
    TRAIN_PATH     = input_dir + "dataset_span_mono.json"
    VAL_PATH       = input_dir + "val_dataset_span_mono.json"
    TEST_PATH      = input_dir + "test_dataset_span_mono.json"
    LABEL2DESC_PATH = input_dir + "label2desc.json"
    LABEL2ID_PATH  = input_dir + "label2id.json"
    MODEL_NAME     = '/kaggle/input/datasets/alessandrobecci/gliner2-1small/'
else:
    input_dir      = "../"
    TRAIN_PATH     = input_dir + "dataset/dataset_span_mono.json"
    VAL_PATH       = input_dir + "dataset/val_dataset_span_mono.json"
    TEST_PATH      = input_dir + "dataset/test_dataset_span_bi.json"
    LABEL2DESC_PATH = input_dir + "dataset/label2desc.json"
    LABEL2ID_PATH  = input_dir + "dataset/label2id.json"
    MODEL_NAME     = "urchade/gliner_small-v2.1"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ MLP PROMPT ENCODER
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
                 hidden_dim=None, dropout=0.1, prompt_len=None,
                 pooling_mode="adaptive_avg", max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout)
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
# 3️⃣ MODELLO BACKBONE
# ==========================================================
print("📦 Caricamento modello...")
model_wrapper = GLiNER.from_pretrained(MODEL_NAME)
model = model_wrapper.model
tokenizer = model_wrapper.data_processor.transformer_tokenizer
backbone = model.token_rep_layer.bert_layer.model
original_word_embeddings = backbone.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim  = original_word_embeddings.embedding_dim

for p in backbone.parameters():
    p.requires_grad = False
print(f"✅ Backbone Frozen. Dim: {embed_dim}")

# ==========================================================
# 4️⃣ LABEL SETUP — CON CLASSE 'O' (nessuna rimozione/rimappatura)
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH)   as f: label2id   = json.load(f)

id2label    = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels  = len(label2id)               # 6 (inclusa O)
O_ID        = label2id.get('O', 5)        # ID della classe O nel vettore delle label

print(f"📋 Label (con O): {label2id}")
print(f"📋 ID classe O: {O_ID}")
print(f"📋 num_labels: {num_labels}")

# Calcolo MAX_TEXT_LEN
MAX_MODEL_LEN       = 512
TOTAL_PROMPT_TOKENS = num_labels * PROMPT_LEN
MAX_TEXT_LEN        = MAX_MODEL_LEN - TOTAL_PROMPT_TOKENS - 5
print(f"📏 Max Text Length: {MAX_TEXT_LEN} (Prompt Tokens: {TOTAL_PROMPT_TOKENS})")

# Tokenize descriptions
desc_texts    = [label2desc[name] for name in label_names]
desc_inputs   = tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids  = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask  = desc_inputs["attention_mask"].to(DEVICE)

prompt_encoder = MLPPromptEncoder(
    original_word_embeddings, vocab_size, embed_dim,
    dropout=DROPOUT_RATE, prompt_len=PROMPT_LEN, pooling_mode=POOLING_MODE,
    max_seq_len=desc_input_ids.shape[1]
).to(DEVICE)

# ==========================================================
# 5️⃣ DATASET — INCLUDE TUTTI GLI SPAN (anche O)
# ==========================================================
class SpanJsonDataset(Dataset):
    """
    Span-level dataset. Include TUTTI gli span — anche quelli con classe O.
    La classe O viene trattata come qualsiasi altra classe.
    """
    def __init__(self, path_json, tokenizer, max_len=512):
        print(f"📖 Leggendo {path_json}...")
        with open(path_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        words = rec["tokenized_text"]
        spans = rec["ner"]   # [[start, end, label_str], ...]

        all_input_ids         = []
        word_to_subword_start = []
        word_to_subword_end   = []

        for word in words:
            subwords = self.tok.encode(word, add_special_tokens=False)
            if not subwords: subwords = [self.tok.unk_token_id]
            word_to_subword_start.append(len(all_input_ids))
            all_input_ids.extend(subwords)
            word_to_subword_end.append(len(all_input_ids) - 1)

        if len(all_input_ids) > self.max_len:
            all_input_ids = all_input_ids[:self.max_len]

        span_labels = []   # [(sub_start, sub_end, label_id)]
        for span in spans:
            word_start, word_end, label_str = span[0], span[1], str(span[2])
            label_id  = int(label_str)   # Usiamo l'ID originale — O inclusa

            sub_start = word_to_subword_start[word_start]
            sub_end   = word_to_subword_end[word_end]

            if sub_start >= len(all_input_ids): continue
            sub_end = min(sub_end, len(all_input_ids) - 1)

            span_labels.append((sub_start, sub_end, label_id))

        return {
            "input_ids":      torch.tensor(all_input_ids),
            "attention_mask": torch.tensor([1] * len(all_input_ids)),
            "span_labels":    span_labels,
        }


def collate_span_batch(batch, pad_id):
    maxlen     = max(len(x["input_ids"]) for x in batch)
    B          = len(batch)
    input_ids  = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask  = torch.zeros((B, maxlen), dtype=torch.long)
    all_spans  = []
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        all_spans.append(ex["span_labels"])
    return {"input_ids": input_ids, "attention_mask": attn_mask, "span_labels": all_spans}


def get_cb_weights_span(dataset_path, label2id, device, beta=0.9999):
    """Calcola pesi Class-Balanced contando TUTTI gli span inclusa O."""
    with open(dataset_path, "r", encoding="utf-8") as f: data = json.load(f)
    label_counts = Counter()
    for record in data:
        for span in record["ner"]:
            label_id = int(str(span[2]))
            label_counts[label_id] += 1

    num_classes = len(label2id)
    weights     = torch.ones(num_classes).to(device)
    print(f"\n  CALCOLO PESI (Class Balanced, Beta {beta}) [ALL SPANS, inclusa O]:")
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else:
            weights[label_id] = 0.0
    weights = weights / weights.sum() * num_classes
    for label_name, label_id in label2id.items():
        print(f"   🔹 {label_name.ljust(15)}: {str(label_counts.get(label_id,0)).rjust(6)} → Peso: {weights[label_id].item():.4f}")
    return weights


print("📊 Loading Dataset...")
train_ds = SpanJsonDataset(TRAIN_PATH, tokenizer, max_len=MAX_TEXT_LEN)
val_ds   = SpanJsonDataset(VAL_PATH,   tokenizer, max_len=MAX_TEXT_LEN)
print(f"🔪 Train: {len(train_ds)} | Val: {len(val_ds)}")

class_weights = get_cb_weights_span(TRAIN_PATH, label2id, DEVICE, beta=CB_BETA)
ce_loss       = FocalLoss(alpha=class_weights, gamma=GAMMA_FOCAL_LOSS, ignore_index=-100)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_span_batch(b, tokenizer.pad_token_id))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=lambda b: collate_span_batch(b, tokenizer.pad_token_id))

# ==========================================================
# 6️⃣ TRAINING LOOP
# ==========================================================
optimizer = optim.AdamW(prompt_encoder.parameters(), lr=LR_MLP, weight_decay=WEIGHT_DECAY)
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(num_training_steps * WARMUP_RATIO),
                                            num_training_steps=num_training_steps)
backbone.to(DEVICE).eval()

def compute_span_loss(text_reps, prompt_vectors, span_labels_batch, loss_fn, temperature, num_labels, O_ID, max_neg_per_sent=15):
    all_logits  = []
    all_targets = []
    H_text    = F.normalize(text_reps,    dim=-1)
    H_prompts = F.normalize(prompt_vectors, dim=-1)
    B = text_reps.shape[0]
    for b in range(B):
        # 1. Gold Spans (Entities + provided O-gaps)
        gold_indices = set()
        for (s_start, s_end, label_id) in span_labels_batch[b]:
            span_vec = H_text[b, s_start:s_end+1, :].mean(dim=0)
            span_vec = F.normalize(span_vec, dim=-1) # Ensure span is on hypersphere
            logits   = torch.mv(H_prompts[b], span_vec) / temperature
            all_logits.append(logits)
            all_targets.append(label_id)
            gold_indices.add((s_start, s_end))
        
        # 2. Random Negative Sampling (improve Precision)
        seq_len = H_text.shape[1]
        if seq_len > 1:
            sampled_neg = 0
            # Proviamo a trovare max_neg_per_sent span casuali che non siano già nei gold
            # Limitiamo la lunghezza dello span campionato a 12 (come nell'estrattore)
            for _ in range(max_neg_per_sent * 4):
                if sampled_neg >= max_neg_per_sent: break
                s = torch.randint(0, seq_len, (1,)).item()
                e = torch.randint(s, min(s + 12, seq_len), (1,)).item()
                if (s, e) not in gold_indices:
                    span_vec = H_text[b, s:e+1, :].mean(dim=0)
                    span_vec = F.normalize(span_vec, dim=-1)
                    logits   = torch.mv(H_prompts[b], span_vec) / temperature
                    all_logits.append(logits)
                    all_targets.append(O_ID)
                    gold_indices.add((s, e))
                    sampled_neg += 1

    if not all_logits:
        return torch.tensor(0.0, device=text_reps.device, requires_grad=True)
    return loss_fn(torch.stack(all_logits), torch.tensor(all_targets, device=text_reps.device, dtype=torch.long))


best_loss    = float('inf')
best_state   = None
patience_ctr = 0
epoch_times  = []

print(f"\n🚀 Training (SPAN-LEVEL, CON O) | LR: {LR_MLP} | Classi: {num_labels}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # --- TRAIN ---
    prompt_encoder.train()
    total_train = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
        ids   = batch["input_ids"].to(DEVICE)
        masks = batch["attention_mask"].to(DEVICE)
        spans = batch["span_labels"]
        optimizer.zero_grad()

        soft_prompts      = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
        soft_prompts_flat = soft_prompts.view(-1, embed_dim)
        prompts_len       = soft_prompts_flat.shape[0]

        cls_tok = torch.tensor([[tokenizer.cls_token_id]] * ids.shape[0], device=DEVICE)
        sep_tok = torch.tensor([[tokenizer.sep_token_id]] * ids.shape[0], device=DEVICE)
        cls_emb = backbone.embeddings(cls_tok)
        sep_emb = backbone.embeddings(sep_tok)
        txt_emb = backbone.embeddings(ids)

        inp_emb = torch.cat([cls_emb, soft_prompts_flat.unsqueeze(0).expand(ids.shape[0], -1, -1),
                             sep_emb, txt_emb, sep_emb], dim=1)
        B = ids.shape[0]
        full_mask = torch.cat([torch.ones((B, 1), device=DEVICE),
                               torch.ones((B, prompts_len), device=DEVICE),
                               torch.ones((B, 1), device=DEVICE),
                               masks, torch.ones((B, 1), device=DEVICE)], dim=1)

        seq_out   = backbone.encoder(inp_emb, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)).last_hidden_state
        text_start = 1 + prompts_len + 1
        text_reps  = seq_out[:, text_start:text_start+ids.shape[1], :]
        pr_seq     = seq_out[:, 1:1+prompts_len, :]
        pr_vecs    = pr_seq.view(B, soft_prompts.shape[0], soft_prompts.shape[1], embed_dim).mean(dim=2)

        loss = compute_span_loss(text_reps, pr_vecs, spans, ce_loss, TEMPERATURE, num_labels, O_ID)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        total_train += loss.item()

    avg_train = total_train / len(train_loader)

    # --- VALIDATION ---
    prompt_encoder.eval()
    total_val = 0.0
    with torch.no_grad():
        for batch in val_loader:
            ids   = batch["input_ids"].to(DEVICE)
            masks = batch["attention_mask"].to(DEVICE)
            spans = batch["span_labels"]

            soft_prompts      = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
            soft_prompts_flat = soft_prompts.view(-1, embed_dim)
            prompts_len       = soft_prompts_flat.shape[0]

            cls_tok = torch.tensor([[tokenizer.cls_token_id]] * ids.shape[0], device=DEVICE)
            sep_tok = torch.tensor([[tokenizer.sep_token_id]] * ids.shape[0], device=DEVICE)
            inp_emb = torch.cat([backbone.embeddings(cls_tok),
                                  soft_prompts_flat.unsqueeze(0).expand(ids.shape[0], -1, -1),
                                  backbone.embeddings(sep_tok),
                                  backbone.embeddings(ids),
                                  backbone.embeddings(sep_tok)], dim=1)
            B = ids.shape[0]
            full_mask = torch.cat([torch.ones((B, 1), device=DEVICE),
                                   torch.ones((B, prompts_len), device=DEVICE),
                                   torch.ones((B, 1), device=DEVICE),
                                   masks, torch.ones((B, 1), device=DEVICE)], dim=1)

            seq_out  = backbone.encoder(inp_emb, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)).last_hidden_state
            text_reps = seq_out[:, 1+prompts_len+1 : 1+prompts_len+1+ids.shape[1], :]
            pr_vecs   = seq_out[:, 1:1+prompts_len, :].view(B, soft_prompts.shape[0], soft_prompts.shape[1], embed_dim).mean(dim=2)
            loss = compute_span_loss(text_reps, pr_vecs, spans, ce_loss, TEMPERATURE, num_labels, O_ID)
            total_val += loss.item()

    avg_val = total_val / len(val_loader)
    elapsed = time.time() - t0
    epoch_times.append({'epoch': epoch, 'train': avg_train, 'val': avg_val, 'time': elapsed})
    print(f"Epoch {epoch}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | {elapsed:.0f}s")

    if avg_val < best_loss:
        best_loss = avg_val
        best_state = {
            'prompt_encoder': {k: v.cpu().clone() for k, v in prompt_encoder.state_dict().items()},
            'config': {
                'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'input_dir': input_dir,
                'train_size': len(train_ds), 'val_size': len(val_ds),
                'random_seed': RANDOM_SEED, 'lr_mlp': LR_MLP, 'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP, 'warmup_ratio': WARMUP_RATIO,
                'temperature': TEMPERATURE, 'dropout_rate': DROPOUT_RATE,
                'weight_strategy': WEIGHT_STRATEGY, 'gamma_focal_loss': GAMMA_FOCAL_LOSS,
                'cb_beta': CB_BETA, 'prompt_len': PROMPT_LEN, 'pooling_mode': POOLING_MODE,
                'max_text_len': MAX_TEXT_LEN, 'patience': EARLY_STOPPING_PATIENCE,
                'loss_level': 'span', 'exclude_O': False,  # O INCLUSA
                'num_labels': num_labels, 'O_ID': O_ID,
            }
        }
        print(f"  → Best model (Val Loss: {best_loss:.4f})")
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early Stopping.")
            break

# ==========================================================
# 7️⃣ SALVATAGGIO
# ==========================================================
from datetime import datetime, timedelta
if best_state is not None:
    os.makedirs("savings", exist_ok=True)
    now = datetime.now()
    if is_running_on_kaggle(): now = now + timedelta(hours=1)
    ts       = now.strftime("%Y%m%d_%H%M%S")
    filename = f"mlp_mono_focal_cbclass_span_withO_val-{ts}.pt"
    torch.save(best_state, os.path.join("savings", filename))
    print(f"💾 Salvato: savings/{filename}")
else:
    print("⚠️ Nessun modello salvato.")

# ==========================================================
# 8️⃣ TEST — CARICAMENTO BEST MODEL E DATI
# ==========================================================
if best_state is None or not os.path.exists(TEST_PATH):
    print("⚠️ Test automatico saltato.")
    exit()

print(f"\n{'='*80}")
print(f"🧪 AVVIO TEST AUTOMATICO (CON E SENZA O)")
print(f"{'='*80}\n")

prompt_encoder.load_state_dict(best_state['prompt_encoder'])
prompt_encoder.eval()

with open(TEST_PATH, 'r', encoding='utf-8') as f: test_data = json.load(f)
print(f"📊 Caricati {len(test_data)} record di test\n")

# Pre-calcola soft prompts fissi
with torch.no_grad():
    soft_prompts      = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
    soft_prompts_flat = soft_prompts.view(-1, embed_dim)
    prompts_len_total = soft_prompts_flat.shape[0]

def forward_sentence(all_inp_ids):
    """Forward pass comune per un singolo record. Ritorna H_text, H_prompts."""
    input_tensor = torch.tensor([all_inp_ids], device=DEVICE)
    attn_mask    = torch.ones_like(input_tensor)
    cls_emb      = backbone.embeddings(torch.tensor([[tokenizer.cls_token_id]], device=DEVICE))
    sep_emb      = backbone.embeddings(torch.tensor([[tokenizer.sep_token_id]], device=DEVICE))
    inp_emb = torch.cat([cls_emb, soft_prompts_flat.unsqueeze(0), sep_emb,
                         backbone.embeddings(input_tensor), sep_emb], dim=1)
    full_mask = torch.ones((1, 1 + prompts_len_total + 1 + len(all_inp_ids) + 1), device=DEVICE)
    seq_out   = backbone.encoder(inp_emb, attention_mask=full_mask.unsqueeze(1).unsqueeze(2)).last_hidden_state
    text_start = 1 + prompts_len_total + 1
    H_text    = F.normalize(seq_out[:, text_start:text_start+len(all_inp_ids), :], dim=-1)
    H_prompts = F.normalize(seq_out[:, 1:1+prompts_len_total, :]
                            .view(1, num_labels, PROMPT_LEN, embed_dim).mean(dim=2), dim=-1)
    return H_text, H_prompts

# ----------------------------------------------------------
# 📋 MODALITÀ CLASSIFICATORE
# ----------------------------------------------------------
print(f"{'='*80}")
print("📋 MODALITÀ CLASSIFICATORE (gold span → argmax tra le classi)")
print(f"{'='*80}\n")

y_true_all, y_pred_all = [], []   # include O
y_true_noO, y_pred_noO = [], []   # esclude O

with torch.no_grad():
    for rec in tqdm(test_data, desc="Classificatore"):
        words = rec["tokenized_text"]
        spans = rec.get("ner", [])

        all_inp_ids, word_to_sw_start, word_to_sw_end = [], [], []
        for word in words:
            subs = tokenizer.encode(word, add_special_tokens=False) or [tokenizer.unk_token_id]
            word_to_sw_start.append(len(all_inp_ids))
            all_inp_ids.extend(subs)
            word_to_sw_end.append(len(all_inp_ids) - 1)

        if len(all_inp_ids) > MAX_TEXT_LEN: all_inp_ids = all_inp_ids[:MAX_TEXT_LEN]
        if not all_inp_ids: continue

        H_text, H_prompts = forward_sentence(all_inp_ids)

        for span in spans:
            w_start, w_end, label_str = span[0], span[1], str(span[2])
            true_label = int(label_str)

            sub_start = word_to_sw_start[w_start]
            sub_end   = word_to_sw_end[w_end]
            if sub_start >= len(all_inp_ids): continue
            sub_end = min(sub_end, len(all_inp_ids) - 1)

            span_vec   = H_text[0, sub_start:sub_end+1, :].mean(dim=0)
            span_vec   = F.normalize(span_vec, dim=-1)
            logits     = torch.mv(H_prompts[0], span_vec)
            pred_label = logits.argmax().item()

            y_true_all.append(true_label)
            y_pred_all.append(pred_label)
            if true_label != O_ID:
                y_true_noO.append(true_label)
                y_pred_noO.append(pred_label)

# Metriche Classificatore
all_ids  = list(range(num_labels))
noO_ids  = [i for i in all_ids if i != O_ID]
noO_names = [label_names[i] for i in noO_ids]

# Con O
mp_all, mr_all, mf1_all, _ = precision_recall_fscore_support(y_true_all, y_pred_all, labels=all_ids, average="macro", zero_division=0)
up_all, ur_all, uf1_all, _ = precision_recall_fscore_support(y_true_all, y_pred_all, labels=all_ids, average="micro", zero_division=0)
# Senza O
mp_noO, mr_noO, mf1_noO, _ = precision_recall_fscore_support(y_true_noO, y_pred_noO, labels=noO_ids, average="macro", zero_division=0)
up_noO, ur_noO, uf1_noO, _ = precision_recall_fscore_support(y_true_noO, y_pred_noO, labels=noO_ids, average="micro", zero_division=0)

print(f"\n🏆 CLASSIFICATORE — Con O (tutte le {num_labels} classi):")
print(f"   MACRO F1: {mf1_all:.4f} | MICRO F1: {uf1_all:.4f}")
print(f"\n🏆 CLASSIFICATORE — Senza O (5 classi reali):")
print(f"   MACRO F1: {mf1_noO:.4f} | MICRO F1: {uf1_noO:.4f}")
print(f"\n📋 Report Dettagliato (senza O):\n")
print(classification_report(y_true_noO, y_pred_noO, labels=noO_ids, target_names=noO_names, zero_division=0))

# ----------------------------------------------------------
# 🔍 MODALITÀ ESTRATTORE (NER reale — proactive scan)
# ----------------------------------------------------------
print(f"\n{'='*80}")
print(f"🔍 MODALITÀ ESTRATTORE (Soglia: {EXTRACTOR_THRESHOLD} | Max span: {EXTRACTOR_MAX_SPAN} parole)")
print(f"   L'O viene usata per SCARTARE gli span non-entità.")
print(f"{'='*80}\n")

tp_dict  = defaultdict(int)
fp_dict  = defaultdict(int)
fn_dict  = defaultdict(int)
sup_dict = defaultdict(int)

with torch.no_grad():
    for rec in tqdm(test_data, desc="Estrattore"):
        words  = rec["tokenized_text"]
        gt_ner = rec.get("ner", [])

        # Gold spans (solo non-O)
        gold_spans = set()
        for (s, e, lbl_str) in gt_ner:
            lbl = int(lbl_str)
            if lbl != O_ID:
                gold_spans.add((s, e, lbl))
                sup_dict[lbl] += 1

        all_inp_ids, word_to_sw_start, word_to_sw_end = [], [], []
        for w in words:
            subs = tokenizer.encode(w, add_special_tokens=False) or [tokenizer.unk_token_id]
            word_to_sw_start.append(len(all_inp_ids))
            all_inp_ids.extend(subs)
            word_to_sw_end.append(len(all_inp_ids) - 1)

        if len(all_inp_ids) > MAX_TEXT_LEN: all_inp_ids = all_inp_ids[:MAX_TEXT_LEN]
        if not all_inp_ids: continue

        H_text, H_prompts = forward_sentence(all_inp_ids)

        # Proactive span scan
        n_words      = len(words)
        candidates   = []
        for start in range(n_words):
            for end in range(start, min(start + EXTRACTOR_MAX_SPAN, n_words)):
                sub_s = word_to_sw_start[start]
                sub_e = word_to_sw_end[end]
                if sub_s >= len(all_inp_ids): break
                sub_e = min(sub_e, len(all_inp_ids) - 1)

                span_vec   = H_text[0, sub_s:sub_e+1, :].mean(dim=0)
                span_vec   = F.normalize(span_vec, dim=-1)
                logits     = torch.mv(H_prompts[0], span_vec)
                best_label = logits.argmax().item()
                best_score = logits[best_label].item()

                # Se la classe migliore NON è O e supera la soglia -> candidato entità
                if best_label != O_ID and best_score > EXTRACTOR_THRESHOLD:
                    candidates.append({'span': (start, end, best_label), 'score': best_score})

        # Greedy NMS
        candidates.sort(key=lambda x: x['score'], reverse=True)
        pred_spans = set()
        occupied   = [False] * n_words
        for cand in candidates:
            s, e, lbl = cand['span']
            if not any(occupied[s:e+1]):
                pred_spans.add((s, e, lbl))
                for i in range(s, e+1): occupied[i] = True

        # TP / FP / FN
        for ps in pred_spans:
            (tp_dict if ps in gold_spans else fp_dict)[ps[2]] += 1
        for gs in gold_spans:
            if gs not in pred_spans: fn_dict[gs[2]] += 1

# Report Estrattore
print(f"\n{'='*80}")
print(f"🎯 RISULTATI ESTRATTORE (Exact Match, solo classi non-O)")
print(f"{'='*80}")
p_list, r_list, f1_list = [], [], []
for lid in noO_ids:
    tp, fp, fn = tp_dict[lid], fp_dict[lid], fn_dict[lid]
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    p_list.append(p); r_list.append(r); f1_list.append(f1)
    print(f"{label_names[lid]:<15} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | Sup: {sup_dict[lid]}")

macro_p  = np.mean(p_list); macro_r = np.mean(r_list); macro_f1 = np.mean(f1_list)
micro_tp = sum(tp_dict[l] for l in noO_ids)
micro_fp = sum(fp_dict[l] for l in noO_ids)
micro_fn = sum(fn_dict[l] for l in noO_ids)
micro_p  = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
micro_r  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

print(f"{'-'*65}")
print(f"MACRO — P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
print(f"MICRO — P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")
print(f"{'='*80}\n")
