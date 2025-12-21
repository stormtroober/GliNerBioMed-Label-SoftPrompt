# -*- coding: utf-8 -*-
"""
TEST MLP Prompt Encoder + Custom Projection.
Carica sia il prompt encoder addestrato SIA la proiezione modificata.
Supporta tutte le tecniche di pooling: adaptive_avg, adaptive_max, attention, conv1d, conv1d_strided
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
TEST_PATH = "../dataset/test_dataset_tokenlevel.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
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
    
    # 2. Crea attention mask per sequenze poolate
    if PROMPT_LEN is not None:
        pooled_attn_mask = torch.ones(soft_embeds.shape[0], soft_embeds.shape[1], 
                                      dtype=torch.long, device=DEVICE)
    else:
        pooled_attn_mask = batch_desc["attention_mask"]
    
    # 3. Label Encoder
    out_lbl = lbl_enc(inputs_embeds=soft_embeds, attention_mask=pooled_attn_mask)
    
    # 4. Pooling + Projection
    mask = pooled_attn_mask.unsqueeze(-1).float()
    pooled = torch.sum(out_lbl.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    label_matrix = F.normalize(proj(pooled), dim=-1)

print(f"✅ Matrice pronta: {label_matrix.shape}")

# ==========================================================
# 5️⃣ TEST LOOP
# ==========================================================
with open(TEST_PATH) as f: test_data = json.load(f)
print(f"\n🔍 Valutazione su {len(test_data)} record...")

y_true, y_pred = [], []

def truncate(tokens):
    if len(tokens) > 512: return tokens[:512]
    return tokens

total_records = len(test_data)
checkpoint_interval = max(1, total_records // 5)

print(f"\n📊 Mostro metriche ogni {checkpoint_interval} record (~20%)\n")

with torch.no_grad():
    for idx, rec in enumerate(tqdm(test_data), 1):
        tokens = truncate(rec["tokens"])
        labels = rec["labels"][:len(tokens)]
        
        inp = txt_tok.convert_tokens_to_ids(tokens)
        inp_tensor = torch.tensor([inp], device=DEVICE)
        mask_tensor = torch.ones_like(inp_tensor)
        
        out_txt = txt_enc(inp_tensor, mask_tensor)
        H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        logits = torch.matmul(H, label_matrix.T).squeeze(0)
        preds = logits.argmax(-1).cpu().tolist()
        
        for p, t in zip(preds, labels):
            if t != -100:
                y_true.append(t)
                y_pred.append(p)
        
        if idx % checkpoint_interval == 0 or idx == total_records:
            if len(y_true) > 0:
                progress = (idx / total_records) * 100
                current_macro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )[2]
                current_micro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, average="micro", zero_division=0
                )[2]
                print(f"\n [{progress:5.1f}%] Macro F1: {current_macro_f1:.4f} | Micro F1: {current_micro_f1:.4f} | Tokens: {len(y_true):,}")

# ==========================================================
# 6️⃣ RISULTATI
# ==========================================================
from sklearn.metrics import confusion_matrix
import datetime

all_label_ids = list(range(len(label_names)))

macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
micro_f1 = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)[2]

per_class_metrics = precision_recall_fscore_support(
    y_true, y_pred, labels=all_label_ids, zero_division=0
)

class_report = classification_report(
    y_true, y_pred, target_names=label_names, labels=all_label_ids, zero_division=0, digits=4
)

pred_counts = Counter(y_pred)
true_counts = Counter(y_true)
conf_matrix = confusion_matrix(y_true, y_pred, labels=all_label_ids)

print(f"\n🏆 MACRO F1: {macro_f1:.4f} | MICRO F1: {micro_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

# ==========================================================
# 📊 EXPORT DETTAGLIATO
# ==========================================================
os.makedirs("test_results", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"test_results/eval_mlp_prompt_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test - MLP Prompt Encoder\n\n")
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
        
        # Descrizione della tecnica di pooling
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
        
        # Ordine preferito per le chiavi importanti
        priority_keys = ['prompt_len', 'pooling_mode', 'batch_size', 'epochs', 'dataset_size',
                        'lr_mlp', 'lr_proj', 'temperature', 'gamma_focal_loss', 'cb_beta']
        
        # Prima le chiavi prioritarie
        for key in priority_keys:
            if key in train_config:
                val = train_config[key]
                if isinstance(val, float) and val < 0.001:
                    val_str = f"{val:.1e}"
                else:
                    val_str = str(val)
                # Evidenzia prompt_len e pooling_mode
                if key in ['prompt_len', 'pooling_mode']:
                    f.write(f"| **{key}** | **{val_str}** |\n")
                else:
                    f.write(f"| {key} | {val_str} |\n")
        
        # Poi le altre chiavi
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

    # Metriche globali
    f.write(f"## 📈 Metriche Globali\n\n")
    f.write(f"| Metric | Score |\n")
    f.write(f"|:-------|------:|\n")
    f.write(f"| **Macro F1** | **{macro_f1:.4f}** |\n")
    f.write(f"| **Micro F1** | **{micro_f1:.4f}** |\n")
    f.write(f"| Token Totali | {len(y_true):,} |\n\n")
    
    # Metriche per classe DETTAGLIATE
    f.write(f"## 📊 Metriche per Classe (Dettagliate)\n\n")
    f.write(f"| Classe | Precision | Recall | F1-Score | Support | Predicted |\n")
    f.write(f"|:-------|----------:|-------:|---------:|--------:|----------:|\n")
    
    precisions, recalls, f1s, supports = per_class_metrics
    for i, label_name in enumerate(label_names):
        pred_count = pred_counts.get(i, 0)
        true_count = supports[i]
        f.write(f"| **{label_name}** | {precisions[i]:.4f} | {recalls[i]:.4f} | "
                f"{f1s[i]:.4f} | {true_count} | {pred_count} |\n")
    
    f.write(f"| **TOTAL** | - | - | - | {sum(supports)} | {len(y_pred)} |\n\n")
    
    f.write(f"## 📋 Classification Report Completo\n\n")
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