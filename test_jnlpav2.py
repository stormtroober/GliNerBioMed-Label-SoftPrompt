# -*- coding: utf-8 -*-
"""
Valutazione token-level del label encoder di GLiNER-BioMed (fine-tuned vs base).
Usa dataset JSON con {tokens, labels}.
"""

import json, torch, torch.nn.functional as F
from gliner import GLiNER
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

# ===============================================================
# ⚙️ CONFIG
# ===============================================================
DATA_JSON = "dataset_test_raw_200.json"
LABEL2ID = "label2id.json"
LABEL2DESC = "label2desc.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ===============================================================
# 📦 CARICAMENTO RISORSE
# ===============================================================
with open(LABEL2ID) as f:
    label2id = json.load(f)
with open(LABEL2DESC) as f:
    label2desc = json.load(f)

id2label = {int(v): k for k, v in label2id.items()}
labels_text = [label2desc[id2label[i]] for i in range(len(label2id))]

with open(DATA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✅ Caricate {len(data)} frasi di test.")

# ===============================================================
# 🧠 MODELLI
# ===============================================================
print("\n📦 Caricamento GLiNER-BioMed base...")
model_base = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core_base = model_base.model
txt_enc = core_base.token_rep_layer.bert_layer.model
lbl_enc_base = core_base.token_rep_layer.labels_encoder.model
proj_base = core_base.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc_base.config._name_or_path)

print("\n🔁 Caricamento label encoder fine-tunato...")
ckpt = torch.load("gliner_label_ft_toklvl_single/final_label_encoder.pt", map_location=device)

model_ft = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core_ft = model_ft.model

# 🔥 carica nello stesso livello usato nel training
core_ft.token_rep_layer.labels_encoder.load_state_dict(ckpt["labels_encoder"])
core_ft.token_rep_layer.labels_projection.load_state_dict(ckpt["labels_projection"])

lbl_enc_ft = core_ft.token_rep_layer.labels_encoder.model   # solo per usarlo dopo
proj_ft = core_ft.token_rep_layer.labels_projection

print("✅ Label encoder e projection caricati correttamente.\n")


# ===============================================================
# 🔧 FUNZIONI
# ===============================================================
def get_label_vecs(desc_texts, lbl_tok, lbl_enc, proj):
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = lbl_enc(**batch).last_hidden_state
    attn = batch["attention_mask"].float().unsqueeze(-1)
    pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
    return F.normalize(proj(pooled), dim=-1)


def evaluate_token_level(name, lbl_enc, proj):
    print(f"🧮 Valutazione token-level per {name}...")
    label_vecs = get_label_vecs(labels_text, lbl_tok, lbl_enc, proj)
    y_true, y_pred = [], []

    for rec in data:
        tokens = rec["tokens"]
        labels = rec["labels"]

        enc = txt_tok(tokens, is_split_into_words=True, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            h_tok = F.normalize(txt_enc(**enc).last_hidden_state.squeeze(0), dim=-1)

        for i, gold_lab in enumerate(labels):
            if gold_lab in ["O", "IGNORE"]:
                continue
            logits = h_tok[i] @ label_vecs.T
            pred_id = logits.argmax().item()
            y_true.append(label2id[gold_lab])
            y_pred.append(pred_id)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    print(f"=== 📈 {name} ===")
    print(f"Macro F1: {f1_macro:.3f} | Micro F1: {f1_micro:.3f}")
    print(f"Precision (macro/micro): {prec_macro:.3f} / {prec_micro:.3f}")
    print(f"Recall (macro/micro): {rec_macro:.3f} / {rec_micro:.3f}\n")

# ===============================================================
# 🧪 TEST DEI MODELLI
# ===============================================================
#evaluate_token_level("GLiNER-BioMed base", lbl_enc_base, proj_base)
evaluate_token_level("GLiNER-BioMed fine-tuned", lbl_enc_ft, proj_ft)

# Dopo aver caricato i due modelli
with torch.no_grad():
    lbl_vecs_base = get_label_vecs(labels_text, lbl_tok, lbl_enc_base, proj_base)
    lbl_vecs_ft = get_label_vecs(labels_text, lbl_tok, lbl_enc_ft, proj_ft)
    cos = torch.nn.functional.cosine_similarity(lbl_vecs_base, lbl_vecs_ft)
    print("Cosine mean:", cos.mean().item(), "min:", cos.min().item(), "max:", cos.max().item())
