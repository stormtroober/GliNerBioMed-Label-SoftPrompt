# -*- coding: utf-8 -*-
"""
TEST BASELINE GLiNER (No Soft Prompt / MLP).
Carica il modello base 'Ihor/gliner-biomed-bi-small-v1.0' e valuta usando solo i nomi delle label.
Serve come baseline per confrontare le performance del modello finetunato.
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
import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "../dataset/test_dataset_tokenlevel.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
OUTPUT_DIR = "test_results"

# ==========================================================
# 1️⃣ CARICAMENTO MODELLO BASE
# ==========================================================
print(f"📦 Caricamento modello base: {MODEL_NAME}...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

# Componenti interni per encoding manuale (per coerenza con lo script di test custom)
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection
txt_enc = core.token_rep_layer.bert_layer.model

# Setup modalità eval
lbl_enc.to(DEVICE).eval()
proj.to(DEVICE).eval()
txt_enc.to(DEVICE).eval()

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

print("✅ Modello caricato e spostato su", DEVICE)

# ==========================================================
# 2️⃣ PRE-CALCOLO MATRICE LABEL (STANDARD MEAN POOLING)
# ==========================================================
print("⚙️  Generazione Matrice Label (Standard con nomi label)...")

# Carica labels
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]

# Tokenizza i NOMI delle label (non descrizioni)
# Standard GLiNER usa [CLS] o Mean Pooling sui nomi. Qui replichiamo la logica base.
batch_lbl = lbl_tok(label_names, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

with torch.no_grad():
    # 1. Encoding
    out_lbl = lbl_enc(input_ids=batch_lbl["input_ids"], attention_mask=batch_lbl["attention_mask"])
    
    # 2. Mean Pooling standard (masked)
    mask = batch_lbl["attention_mask"].unsqueeze(-1).float()
    pooled = torch.sum(out_lbl.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    # 3. Projection & Normalize
    label_matrix = F.normalize(proj(pooled), dim=-1)

print(f"✅ Matrice pronta: {label_matrix.shape}")

# ==========================================================
# 3️⃣ TEST LOOP
# ==========================================================
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Non trovo il dataset di test: {TEST_PATH}")

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
                print(f" [{progress:5.1f}%] Macro F1: {current_macro_f1:.4f} | Micro F1: {current_micro_f1:.4f} | Tokens: {len(y_true):,}")

# ==========================================================
# 4️⃣ RISULTATI & REPORT
# ==========================================================
all_label_ids = list(range(len(label_names)))

macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
micro_f1 = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)[2]
class_report = classification_report(
    y_true, y_pred, target_names=label_names, labels=all_label_ids, zero_division=0, digits=4
)
pred_counts = Counter(y_pred)
true_counts = Counter(y_true)

print(f"\n🏆 BASELINE RESULTS - MACRO F1: {macro_f1:.4f} | MICRO F1: {micro_f1:.4f}")
print(class_report)

# Export Report
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{OUTPUT_DIR}/eval_baseline_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test BASELINE (No Fine-tuning)\n\n")
    f.write(f"**Modello:** `{MODEL_NAME}`\n\n")
    f.write(f"**Dataset:** `{TEST_PATH}`\n\n")
    f.write(f"**Label Encoding:** Nomi Label (Standard Mean Pooling)\n\n")
    f.write(f"**Data:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## 📈 Metriche Globali\n\n")
    f.write(f"| Metric | Score |\n")
    f.write(f"|:-------|------:|\n")
    f.write(f"| **Macro F1** | **{macro_f1:.4f}** |\n")
    f.write(f"| **Micro F1** | **{micro_f1:.4f}** |\n")
    f.write(f"| Token Totali | {len(y_true):,} |\n\n")
    
    f.write(f"## 📋 Classification Report\n\n")
    f.write(f"```\n{class_report}\n```\n\n")
    
    # Distribuzione veloce
    f.write(f"## 🔢 Distribuzione Predizioni\n\n")
    f.write(f"| Classe | Predette | Vere | Diff |\n")
    f.write(f"|:-------|:--------:|:----:|:---:|\n")
    for i in sorted(pred_counts.keys(), key=lambda x: true_counts.get(x, 0), reverse=True):
        label_name = label_names[i]
        pred = pred_counts[i]
        true = true_counts.get(i, 0)
        f.write(f"| {label_name} | {pred} | {true} | {pred - true:+} |\n")

print(f"\n💾 Report salvato: {filename}")
