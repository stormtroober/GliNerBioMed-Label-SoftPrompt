# -*- coding: utf-8 -*-
"""
Confronto tra GLiNER-BioMed originale e il modello fine-tunato salvato.
Valuta Precision / Recall / F1 sugli stessi dati di training.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import json
from gliner import GLiNER
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

# =====================================================
# 1️⃣ Carica configurazione e dati
# =====================================================
print("📥 Caricamento label2desc / label2id / train_data_balanced.csv...")

with open("label2desc.json") as f:
    label2desc = json.load(f)
with open("label2id.json") as f:
    label2id = json.load(f)

labels = list(label2desc.keys())
desc_texts = [label2desc[l] for l in labels]
id2label = {int(v): k for k, v in label2id.items()}

train_df = pd.read_csv("test_data_random.csv")
train_data = [(row["text"], eval(row["entity_span"]), int(row["label_id"])) for _, row in train_df.iterrows()]

# =====================================================
# 2️⃣ Funzioni comuni
# =====================================================
def get_span_vec_tokens(text, token_span, tokenizer, encoder):
    s_word, e_word = token_span
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", add_special_tokens=True)
    word_ids = enc.word_ids(batch_index=0)
    sw = next(i for i, w in enumerate(word_ids) if w == s_word)
    ew = max(i for i, w in enumerate(word_ids) if w == e_word)
    with torch.no_grad():
        hidden = encoder(**{k: enc[k] for k in ["input_ids", "attention_mask"]}).last_hidden_state.squeeze(0)
    return hidden[sw:ew+1].mean(dim=0)

def get_label_vecs(labels, lbl_tok, lbl_enc, proj):
    batch = lbl_tok(labels, return_tensors="pt", padding=True, truncation=True)
    out = lbl_enc(**batch)
    attn = batch["attention_mask"].float().unsqueeze(-1)
    he_384 = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
    return proj(he_384)

# =====================================================
# 3️⃣ Carica modello BASE
# =====================================================
print("\n🧠 Caricamento modello base GLiNER-BioMed...")
model_base = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core_base = model_base.model
txt_enc = core_base.token_rep_layer.bert_layer.model
lbl_enc = core_base.token_rep_layer.labels_encoder.model
proj = core_base.token_rep_layer.labels_projection

from transformers import AutoTokenizer
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# =====================================================
# 4️⃣ Carica modello FINE-TUNED
# =====================================================
print("🔁 Caricamento modello fine-tunato da ./trained_gliner_model ...")

# carica modello nuovo con stessi pesi iniziali
model_ft = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core_ft = model_ft.model

checkpoint = torch.load("trained_gliner_model/training_checkpoint.pt", map_location="cpu")
weights = torch.load("trained_gliner_model/model_weights.pt", map_location="cpu")

core_ft.load_state_dict(weights, strict=False)
core_ft.token_rep_layer.labels_encoder.load_state_dict(checkpoint["labels_encoder"])
core_ft.token_rep_layer.labels_projection.load_state_dict(checkpoint["labels_projection"])

lbl_enc_ft = core_ft.token_rep_layer.labels_encoder.model
proj_ft = core_ft.token_rep_layer.labels_projection

# =====================================================
# 5️⃣ Valutazione dei due modelli
# =====================================================
def evaluate_model(lbl_enc, proj, name):
    print(f"🧮 Precomputing label embeddings for {name}...")
    label_vecs = get_label_vecs(desc_texts, lbl_tok, lbl_enc, proj)
    label_vecs = F.normalize(label_vecs, dim=-1)  # normalizza una sola volta
    
    y_true, y_pred = [], []
    for text, token_span, gold_id in train_data:
        span_vec = get_span_vec_tokens(text, token_span, txt_tok, txt_enc)
        span_vec = F.normalize(span_vec, dim=-1)
        logits = span_vec @ label_vecs.T
        pred_id = logits.argmax().item()
        y_true.append(gold_id)
        y_pred.append(pred_id)

    all_label_ids = list(range(len(labels)))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=all_label_ids
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0, labels=all_label_ids
    )
    print(f"\n=== 📈 {name} ===")
    print(f"Macro F1: {f1_macro:.3f} | Micro F1: {f1_micro:.3f}")
    print(f"Precision (macro/micro): {prec_macro:.3f} / {prec_micro:.3f}")
    print(f"Recall (macro/micro): {rec_macro:.3f} / {rec_micro:.3f}")

    if(name == "GLiNER-BioMed fine-tuned"):
        filename = "fine_tuned_results.md"
    else:
        filename = "base_model_results.md"

    with open(filename, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Macro F1: {f1_macro:.3f} | Micro F1: {f1_micro:.3f}\n")
        f.write(f"Precision (macro/micro): {prec_macro:.3f} / {prec_micro:.3f}\n")
        f.write(f"Recall (macro/micro): {rec_macro:.3f} / {rec_micro:.3f}\n")    
        
    return y_true, y_pred

# =====================================================
# 6️⃣ Confronto
# =====================================================
print("\n🚀 Valutazione del modello base...")
evaluate_model(lbl_enc, proj, "GLiNER-BioMed base")

print("\n🚀 Valutazione del modello fine-tunato...")
evaluate_model(lbl_enc_ft, proj_ft, "GLiNER-BioMed fine-tuned")

