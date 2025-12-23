# -*- coding: utf-8 -*-
"""
TEST MLP Prompt Encoder + Custom Projection (NO 'O' CLASS IN METRICS).
Carica sia il prompt encoder addestrato SIA la proiezione modificata.
Supporta tutte le tecniche di pooling: adaptive_avg, adaptive_max, attention, conv1d, conv1d_strided
NOTA: In questo script, la classe 'O' viene ESCLUSA dal calcolo delle metriche F1 (Macro/Micro) e dal report.
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from torch import nn
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter
import os
from tqdm import tqdm
import subprocess

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
prefix = "../dataset/"
TEST_PATH = prefix + "test_dataset_tknlvl_bi.json"
LABEL2DESC_PATH = prefix + "label2desc.json"
LABEL2ID_PATH = prefix + "label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# 1️⃣ RIDEFINIZIONE CLASSI (Identiche al Training)
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
        # Non serve copiare i pesi qui, tanto li sovrascriviamo col load_state_dict
        
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
            self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode, max_seq_len=max_seq_len)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        
        # Applica pooling se configurato
        if self.pooler is not None:
            x = self.pooler(x, attention_mask)
        
        return x

# ==========================================================
# 2️⃣ SELEZIONE CHECKPOINT
# ==========================================================
def select_checkpoint_interactive():
    if not os.path.exists(SAVINGS_DIR): return None
    ckpts = [f for f in os.listdir(SAVINGS_DIR) if f.endswith('.pt')]
    if not ckpts: return None
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(SAVINGS_DIR, x)), reverse=True)
    
    print("\n📦 CHECKPOINT DISPONIBILI:")
    for i, c in enumerate(ckpts[:10]): print(f"{i+1}. {c}")
    try:
        sel = int(input("\n👉 Scegli numero [1]: ") or 1) - 1
        return os.path.join(SAVINGS_DIR, ckpts[sel])
    except: return os.path.join(SAVINGS_DIR, ckpts[0])

ckpt_path = select_checkpoint_interactive()
print(f"Loading: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

train_config = checkpoint.get('config', {})
if train_config:
    print("✅ Configurazione training trovata.")
else:
    print("⚠️ Nessuna configurazione trovata nel checkpoint.")

# Deduzione presenza projection addestrata
projection_trained = 'projection' in checkpoint
print(f"{'✅' if projection_trained else '⚠️'} Proiezione custom {'presente' if projection_trained else 'NON presente'}")

# ⚠️ CRITICAL: Recupera configurazione pooling dal checkpoint
PROMPT_LEN = train_config.get('prompt_len', None)
POOLING_MODE = train_config.get('pooling_mode', 'adaptive_avg')

print(f"\n{'='*60}")
print(f"🎯 CONFIGURAZIONE PROMPT POOLING RILEVATA:")
print(f"{'='*60}")
if PROMPT_LEN is not None:
    print(f"   ✅ Prompt Length: {PROMPT_LEN}")
    print(f"   ✅ Pooling Mode:  {POOLING_MODE}")
else:
    print(f"   ⚠️  Nessun pooling (full sequence length)")
print(f"{'='*60}\n")

# ==========================================================
# 3️⃣ CARICAMENTO MODELLO IBRIDO
# ==========================================================
print("📦 Ricostruzione Architettura...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

# A. Carichiamo la PROIEZIONE ADDESTRATA
if 'projection' in checkpoint:
    proj.load_state_dict(checkpoint['projection'])
    print("✅ Proiezione custom caricata.")
else:
    print("⚠️ ATTENZIONE: 'projection' non trovata nel checkpoint. Usando quella originale.")

# B. Carichiamo il PROMPT ENCODER con configurazione pooling corretta
vocab_size = lbl_enc.embeddings.word_embeddings.num_embeddings
embed_dim = lbl_enc.embeddings.word_embeddings.embedding_dim

# 🔥 CRITICAL: Crea prompt encoder CON la configurazione di pooling dal checkpoint
prompt_encoder = MLPPromptEncoder(
    lbl_enc.embeddings.word_embeddings, 
    vocab_size, 
    embed_dim,
    prompt_len=PROMPT_LEN,
    pooling_mode=POOLING_MODE
).to(DEVICE)

if 'prompt_encoder' in checkpoint:
    prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
    print(f"✅ Prompt Encoder caricato (Pooling: {POOLING_MODE if PROMPT_LEN else 'None'}).")
elif 'soft_embeddings' in checkpoint:
    pass
else:
    try:
        prompt_encoder.load_state_dict(checkpoint)
        print("✅ Prompt Encoder caricato (direct state dict).")
    except:
        print("❌ Errore caricamento Prompt Encoder.")

# Setup modalità eval
txt_enc = core.token_rep_layer.bert_layer.model.to(DEVICE).eval()
lbl_enc.to(DEVICE).eval()
proj.to(DEVICE).eval()
prompt_encoder.eval()

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# ==========================================================
# 4️⃣ PRE-CALCOLO MATRICE LABEL (CON POOLING CORRETTO!)
# ==========================================================
print("⚙️  Generazione Matrice Label (con pooling identico al training)...")
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]

desc_texts = [label2desc[name] for name in label_names]
batch_desc = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

with torch.no_grad():
    # 1. MLP Transform + POOLING
    soft_embeds = prompt_encoder(batch_desc["input_ids"], attention_mask=batch_desc["attention_mask"])
    
    print(f"   • Shape dopo MLP+Pooling: {soft_embeds.shape}")

    # --- NEW LOGIC: [CLS] + MLP_PROMPTS + [SEP] (TESTING) ---
    # IDs per CLS e SEP
    cls_id = torch.tensor([[lbl_tok.cls_token_id]], device=DEVICE) 
    sep_id = torch.tensor([[lbl_tok.sep_token_id]], device=DEVICE)
    
    # Espandi per il numero di label (batch size attuale)
    num_labels_batch = soft_embeds.shape[0]
    
    cls_embeds = lbl_enc.embeddings.word_embeddings(cls_id).expand(num_labels_batch, -1, -1) 
    sep_embeds = lbl_enc.embeddings.word_embeddings(sep_id).expand(num_labels_batch, -1, -1) 
    
    # Concatena
    enclosed_soft_embeds = torch.cat([cls_embeds, soft_embeds, sep_embeds], dim=1) 

    # 2. Crea attention mask
    if PROMPT_LEN is not None:
        total_len = 1 + soft_embeds.shape[1] + 1
        pooled_attn_mask = torch.ones(num_labels_batch, total_len, dtype=torch.long, device=DEVICE)
    else:
        # Fallback (non dovrebbe accadere se config da training è corretta)
        total_len = 1 + soft_embeds.shape[1] + 1
        pooled_attn_mask = torch.ones(num_labels_batch, total_len, dtype=torch.long, device=DEVICE)
    
    # 3. Label Encoder
    out_lbl = lbl_enc(inputs_embeds=enclosed_soft_embeds, attention_mask=pooled_attn_mask)
    
    # 4. Pooling + Projection
    mask = pooled_attn_mask.unsqueeze(-1).float()
    pooled = torch.sum(out_lbl.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    label_matrix = F.normalize(proj(pooled), dim=-1)

print(f"✅ Matrice pronta: {label_matrix.shape}")

# ==========================================================
# 5️⃣ TEST LOOP (BATCHED)
# ==========================================================
BATCH_SIZE = 16
with open(TEST_PATH) as f: test_data = json.load(f)
print(f"\n🔍 Valutazione su {len(test_data)} record (Batch Size: {BATCH_SIZE})...")

y_true, y_pred = [], []

def truncate(tokens):
    if len(tokens) > 512: return tokens[:512]
    return tokens

# Helper per batching
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

total_records = len(test_data)
checkpoint_interval = max(1, total_records // 5)

print(f"\n📊 Mostro metriche ogni {checkpoint_interval} record (~20%)\n")

# IDENTIFICAZIONE CLASSE 'O'
ignore_index = -1
for idx, name in enumerate(label_names):
    if name == 'O':
        ignore_index = idx
        break

all_label_ids = list(range(len(label_names)))
if ignore_index != -1:
    relevant_label_ids = [i for i in all_label_ids if i != ignore_index]
    relevant_label_names = [label_names[i] for i in relevant_label_ids]
    print(f"ℹ️  Esclusione classe 'O' (ID: {ignore_index}) dalle metriche aggregate.")
else:
    relevant_label_ids = all_label_ids
    relevant_label_names = label_names
    print("⚠️  Classe 'O' non trovata. Calcolo standard su tutte le classi.")

processed_count = 0

with torch.no_grad():
    for batch_idx, batch_data in enumerate(tqdm(chunked(test_data, BATCH_SIZE), total=(len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE)):
        
        # Prepare Batch
        batch_tokens = []
        batch_labels = []
        
        for rec in batch_data:
            toks = truncate(rec["tokens"])
            batch_tokens.append(toks)
            # Taglio labels se necessario
            batch_labels.append(rec["labels"][:len(toks)])
        
        # Tokenizzazione con Padding (gestita dal tokenizer o manualmente)
        # Qui usiamo txt_tok per convertire e pad manually o via call
        # txt_tok.pad_token_id deve essere definito
        
        # Convert to IDs
        batch_input_ids = [txt_tok.convert_tokens_to_ids(t) for t in batch_tokens]
        
        # Pad
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_masks = []
        
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids = ids + [txt_tok.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_input_ids.append(padded_ids)
            padded_masks.append(mask)
            
        inp_tensor = torch.tensor(padded_input_ids, device=DEVICE)
        mask_tensor = torch.tensor(padded_masks, device=DEVICE)
        
        # Forward Pass
        out_txt = txt_enc(inp_tensor, mask_tensor)
        H = F.normalize(out_txt.last_hidden_state, dim=-1) # (B, SeqLen, D)
        
        # Logits Calculation: (B, SeqLen, D) @ (NumLabels, D)^T -> (B, SeqLen, NumLabels)
        logits = torch.matmul(H, label_matrix.T) 
        
        # Argmax Predictions
        batch_preds = logits.argmax(-1).cpu().tolist() # (B, SeqLen)
        
        # Collect Results (ignoring padding and -100 labels)
        for i, preds_seq in enumerate(batch_preds):
            original_len = len(batch_input_ids[i]) # lunghezza reale senza padding
            valid_preds = preds_seq[:original_len]
            valid_labels = batch_labels[i]
            
            for p, t in zip(valid_preds, valid_labels):
                if t != -100:
                    y_true.append(t)
                    y_pred.append(p)
        
        processed_count += len(batch_data)
        
        # Logging Intermedio
        if processed_count >= total_records or (processed_count % checkpoint_interval < BATCH_SIZE):
             if len(y_true) > 0:
                progress = (processed_count / total_records) * 100
                current_macro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
                )[2]
                current_micro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
                )[2]
                # Print sovrascrive tqdm a volte, usiamo write
                tqdm.write(f" [{progress:5.1f}%] Macro F1 (No O): {current_macro_f1:.4f} | Micro F1: {current_micro_f1:.4f}")

# ==========================================================
# 6️⃣ RISULTATI
# ==========================================================
from sklearn.metrics import confusion_matrix
import datetime

# ==========================================================
# 6️⃣ RISULTATI
# ==========================================================
from sklearn.metrics import confusion_matrix
import datetime

# --- Metriche Globali (ESCLUDENDO O) ---
# Macro Average
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
)

# Micro Average
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
)

# Weighted Average (opzionale, ma utile)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="weighted", zero_division=0
)

# Metriche per classe (TUTTE LE CLASSI, INCLUSA O, per dettaglio)
per_class_metrics = precision_recall_fscore_support(
    y_true, y_pred, labels=all_label_ids, zero_division=0
)

# Report di classificazione (ESCLUDENDO O)
class_report = classification_report(
    y_true, y_pred, target_names=relevant_label_names, labels=relevant_label_ids, zero_division=0, digits=4
)

pred_counts = Counter(y_pred)
true_counts = Counter(y_true)
conf_matrix = confusion_matrix(y_true, y_pred, labels=all_label_ids)

print(f"\n🏆 GLOBAL METRICS (No O Class):")
print(f"   • MACRO:    Precision={macro_p:.4f} | Recall={macro_r:.4f} | F1={macro_f1:.4f}")
print(f"   • MICRO:    Precision={micro_p:.4f} | Recall={micro_r:.4f} | F1={micro_f1:.4f}")
print(f"   • WEIGHTED: Precision={weighted_p:.4f} | Recall={weighted_r:.4f} | F1={weighted_f1:.4f}")

print("\n📋 Classification Report (No O):")
print(classification_report(y_true, y_pred, target_names=relevant_label_names, labels=relevant_label_ids, zero_division=0))

# ==========================================================
# 📊 EXPORT DETTAGLIATO
# ==========================================================
os.makedirs("test_results", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"test_results/eval_mlp_prompt_no_O_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test (NO 'O' CLASS) - MLP Prompt Encoder\n\n")
    f.write(f"> ℹ️ **NOTA**: Le metriche Globali e il Report di Classificazione ESCLUDONO la classe 'O' (Non-classe).\n\n")
    f.write(f"**Checkpoint:** `{os.path.basename(ckpt_path)}`\n\n")
    f.write(f"**Dataset:** `{TEST_PATH}`\n\n")
    f.write(f"**Data:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # ========================================
    # 🎯 SEZIONE PROMINENTE: PROMPT CONFIGURATION
    # ========================================
    f.write(f"---\n\n")
    f.write(f"## 🎯 CONFIGURAZIONE PROMPT TUNING\n\n")
    f.write(f"| Parametro | Valore |\n")
    f.write(f"|:----------|:-------|\n")
    
    if PROMPT_LEN is not None:
        f.write(f"| **🔧 Prompt Length** | **{PROMPT_LEN}** |\n")
        f.write(f"| **🔧 Pooling Mode** | **{POOLING_MODE}** |\n")
        
        pooling_descriptions = {
            "adaptive_avg": "Media adattiva sulla sequenza",
            "adaptive_max": "Max pooling adattivo sulla sequenza", 
            "attention": "Cross-attention con query learnable",
            "conv1d": "Conv1D + AdaptiveAvgPool",
            "conv1d_strided": "Conv1D con gating mechanism"
        }
        desc = pooling_descriptions.get(POOLING_MODE, "Sconosciuto")
        f.write(f"| **📝 Descrizione** | {desc} |\n")
    else:
        f.write(f"| **🔧 Prompt Length** | ❌ Nessun pooling (full length) |\n")
        f.write(f"| **🔧 Pooling Mode** | N/A |\n")
    
    f.write(f"| **Projection Trained** | {'✅ Sì' if projection_trained else '❌ No (originale)'} |\n")
    f.write(f"\n---\n\n")
    
    # ========================================
    # ⚙️ PARAMETRI DI TRAINING COMPLETI
    # ========================================
    if train_config:
        f.write(f"## ⚙️ Parametri di Training Salvati\n\n")
        f.write(f"| Parametro | Valore |\n")
        f.write(f"|:----------|:-------|\n")
        
        priority_keys = ['prompt_len', 'pooling_mode', 'batch_size', 'epochs', 'dataset_size',
                        'lr_mlp', 'lr_proj', 'temperature', 'gamma_focal_loss', 'cb_beta']
        
        for key in priority_keys:
            if key in train_config:
                val = train_config[key]
                if isinstance(val, float) and val < 0.001:
                    val_str = f"{val:.1e}"
                else:
                    val_str = str(val)
                if key in ['prompt_len', 'pooling_mode']:
                    f.write(f"| **{key}** | **{val_str}** |\n")
                else:
                    f.write(f"| {key} | {val_str} |\n")
        
        for key in sorted(train_config.keys()):
            if key not in priority_keys:
                val = train_config[key]
                if isinstance(val, float) and val < 0.001:
                    val_str = f"{val:.1e}"
                else:
                    val_str = str(val)
                f.write(f"| {key} | {val_str} |\n")
        f.write("\n")
    else:
        f.write("> ⚠️ Configurazione di training non presente in questo checkpoint.\n\n")

    # Metriche globali (NO O)
    f.write(f"## 📈 Metriche Globali (ESCLUSO 'O')\n\n")
    
    f.write(f"### Riassunto Performance\n")
    f.write(f"| Average Type | Precision | Recall | F1-Score |\n")
    f.write(f"|:-------------|----------:|-------:|---------:|\n")
    f.write(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |\n")
    f.write(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |\n")
    f.write(f"| Weighted     | {weighted_p:.4f} | {weighted_r:.4f} | {weighted_f1:.4f} |\n\n")
    
    f.write(f"**Token Totali Valutati**: {len(y_true):,}\n\n")
    
    # Metriche per classe (TUTTE, per completezza)
    f.write(f"## 📊 Metriche per Classe (Tutte incluse)\n\n")
    f.write(f"| Classe | Precision | Recall | F1-Score | Support | Predicted |\n")
    f.write(f"|:-------|----------:|-------:|---------:|--------:|----------:|\n")
    
    precisions, recalls, f1s, supports = per_class_metrics
    for i, label_name in enumerate(label_names):
        pred_count = pred_counts.get(i, 0)
        true_count = supports[i]
        
        # Evidenzia se è la classe 'O'
        prefix = "**" if i == ignore_index else ""
        suffix = "** (Esclusa dal Macro/Micro)" if i == ignore_index else ""
        
        f.write(f"| {prefix}{label_name}{suffix} | {precisions[i]:.4f} | {recalls[i]:.4f} | "
                f"{f1s[i]:.4f} | {true_count} | {pred_count} |\n")
    
    f.write(f"| **TOTAL** | - | - | - | {sum(supports)} | {len(y_pred)} |\n\n")
    
    f.write(f"## 📋 Classification Report (No 'O')\n\n")
    f.write(f"```\n{class_report}\n```\n\n")
    
    # Distribuzione predizioni vs ground truth
    f.write(f"## 🔢 Distribuzione Predizioni vs Ground Truth\n\n")
    f.write(f"| Classe | Predette | Vere | Differenza | % Coverage |\n")
    f.write(f"|:-------|:--------:|:----:|:----------:|:----------:|\n")
    
    for i in sorted(pred_counts.keys(), key=lambda x: true_counts.get(x, 0), reverse=True):
        label_name = label_names[i]
        pred = pred_counts[i]
        true = true_counts.get(i, 0)
        diff = pred - true
        coverage = (pred / true * 100) if true > 0 else 0
        f.write(f"| {label_name} | {pred} | {true} | {diff:+d} | {coverage:.1f}% |\n")
    
    f.write(f"\n")
    
    # Matrice di confusione (top confusions)
    f.write(f"## 🔀 Top Confusioni (Errori più frequenti)\n\n")
    f.write(f"| Vera Classe | Predetta Come | Count |\n")
    f.write(f"|:------------|:--------------|------:|\n")
    
    confusions = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and conf_matrix[i][j] > 0:
                confusions.append((label_names[i], label_names[j], conf_matrix[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_label, pred_label, count in confusions[:20]:
        f.write(f"| {true_label} | {pred_label} | {count} |\n")

print(f"\n💾 Report dettagliato salvato: {filename}")

SHELL_SCRIPT_NAME = "orderTests.sh"

print(f"\n🚀 Esecuzione script shell: {SHELL_SCRIPT_NAME}...")

if not os.path.isfile(SHELL_SCRIPT_NAME):
    print(f"❌ ERRORE: Il file '{SHELL_SCRIPT_NAME}' non esiste nella directory corrente.")
else:
    try:
        subprocess.run(["bash", SHELL_SCRIPT_NAME], check=True)
        print("\n✅ Script shell completato con successo.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERRORE CRITICO: Lo script shell è terminato con codice {e.returncode}.")
    except Exception as e:
        print(f"\n❌ ERRORE IMPREVISTO: {e}")
