# -*- coding: utf-8 -*-
"""
Test modello base GLiNER-BioMed con LABEL NAMES
================================================
Valuta il modello base usando i nomi delle label invece delle descrizioni.

Questo test serve per capire la differenza di performance tra:
- Usare nomi brevi: "cell type", "rna", "protein", "dna", "cell line", "O"
- Usare descrizioni lunghe (label2desc.json)

Test set: test_dataset_tokenlevel.json
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter
import os
from datetime import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "../dataset/test_dataset_tknlvl_bi.json"
LABEL2ID_PATH = "../dataset/label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

print(f"🖥️  Device: {DEVICE}")
print(f"📅 Test eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==========================================================
# 1️⃣ CARICAMENTO LABELS E TEST SET
# ==========================================================
print("\n📥 Caricamento configurazione...")
with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
label_names = list(label2id.keys())

# Label names come testo per embedding (NO descrizioni)
label2name = {label: label for label in label_names}

print(f"📏 Label Names usate per embedding:")
for label in label_names:
    print(f"   • {label}")

print(f"\n📚 Caricamento test set da {TEST_PATH}...")
with open(TEST_PATH, "r") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

# ==========================================================
# 2️⃣ FUNZIONI HELPER
# ==========================================================
def compute_label_matrix_from_names(label_names, lbl_tok, lbl_enc, proj):
    """
    Embedda i NOMI delle label (non le descrizioni) con encoder + projection.
    Usa direttamente: "cell type", "rna", "protein", ecc.
    """
    # Usa i nomi delle label come testo
    name_texts = label_names  # ["cell type", "rna", "cell line", "dna", "protein", "O"]
    
    batch = lbl_tok(name_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)


def evaluate_model_with_label_names(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name):
    """Valuta il modello sul test set usando i NOMI delle label."""
    print(f"\n🔍 Valutazione {model_name} con LABEL NAMES...")
    
    txt_enc.eval()
    lbl_enc.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        # Precomputa label embeddings usando i NOMI
        label_matrix = compute_label_matrix_from_names(label_names, lbl_tok, lbl_enc, proj).to(DEVICE)
        
        for record in test_records:
            # Ricrea sempre input_ids e attention_mask dai tokens
            labels = record["labels"]
            tokens = record["tokens"]  # includono [CLS]/[SEP]

            ids = txt_tok.convert_tokens_to_ids(tokens)

            # Troncamento sicuro su max length del modello
            max_len = getattr(txt_tok, "model_max_length", 0) or 0
            if max_len > 0 and len(ids) > max_len:
                if len(ids) >= 2 and max_len >= 2:
                    middle_budget = max_len - 2
                    ids = [ids[0]] + ids[1:1+middle_budget] + [ids[-1]]
                else:
                    ids = ids[:max_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)
            labels = record["labels"]
            
            # Forward pass
            out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            # Similarità token-label
            logits = torch.matmul(H, label_matrix.T).squeeze(0)
            preds = logits.argmax(-1).cpu().numpy()
            
            # Raccogli solo token validi (non -100)
            for pred, true_label in zip(preds, labels):
                if true_label != -100:
                    y_true_all.append(true_label)
                    y_pred_all.append(pred)
    
    # Calcola metriche
    all_label_ids = list(range(len(label_names)))
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0, labels=all_label_ids
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="micro", zero_division=0, labels=all_label_ids
    )
    
    # Report per classe
    class_report = classification_report(
        y_true_all, y_pred_all, 
        target_names=label_names,
        labels=all_label_ids,
        zero_division=0
    )
    
    # Stampa risultati
    print(f"\n{'='*60}")
    print(f"📊 RISULTATI - {model_name} (con Label Names)")
    print(f"{'='*60}")
    print(f"Totale token valutati: {len(y_true_all)}")
    print(f"\n🎯 METRICHE AGGREGATE:")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"  Micro F1:  {f1_micro:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro):    {rec_macro:.4f}")
    print(f"  Precision (micro): {prec_micro:.4f}")
    print(f"  Recall (micro):    {rec_micro:.4f}")
    
    print(f"\n📋 REPORT PER CLASSE:")
    print(class_report)
    
    # Distribuzione predizioni vs ground truth
    pred_counts = Counter(y_pred_all)
    true_counts = Counter(y_true_all)
    
    print(f"\n📈 DISTRIBUZIONE (Top 6):")
    print(f"{'Label':<15} {'Pred':<8} {'True':<8}")
    print("-" * 35)
    for label_id in sorted(pred_counts.keys(), key=lambda x: pred_counts[x], reverse=True)[:6]:
        label_name = id2label[label_id]
        print(f"{label_name:<15} {pred_counts[label_id]:<8} {true_counts.get(label_id, 0):<8}")
    
    # Salva risultati
    os.makedirs("testresults", exist_ok=True)
    filename = f"results_{model_name.replace(' ', '_').replace('-', '_')}_LabelNames.md"
    with open(f'testresults/{filename}', "w") as f:
        f.write(f"# Risultati - {model_name} (con Label Names)\n\n")
        f.write(f"**Data test:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Tipo di embedding label:** Nomi delle label (non descrizioni)\n")
        f.write(f"**Labels usate:** {', '.join(label_names)}\n\n")
        f.write(f"**Token valutati:** {len(y_true_all)}\n\n")
        f.write(f"## Metriche aggregate\n\n")
        f.write(f"- **Macro F1:** {f1_macro:.4f}\n")
        f.write(f"- **Micro F1:** {f1_micro:.4f}\n")
        f.write(f"- **Precision (macro):** {prec_macro:.4f}\n")
        f.write(f"- **Recall (macro):** {rec_macro:.4f}\n")
        f.write(f"- **Precision (micro):** {prec_micro:.4f}\n")
        f.write(f"- **Recall (micro):** {rec_micro:.4f}\n\n")
        f.write(f"## Report per classe\n\n```\n{class_report}\n```\n")
    
    print(f"\n💾 Risultati salvati in: testresults/{filename}")
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro
    }


# ==========================================================
# 3️⃣ CARICAMENTO E VALUTAZIONE MODELLO BASE
# ==========================================================
print("\n" + "="*60)
print("🔹 MODELLO BASE (con Label Names)")
print("="*60)
print(f"📦 Caricamento modello: {MODEL_NAME}")

model_base = GLiNER.from_pretrained(MODEL_NAME)
core_base = model_base.model

txt_enc_base = core_base.token_rep_layer.bert_layer.model
lbl_enc_base = core_base.token_rep_layer.labels_encoder.model
proj_base = core_base.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc_base.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc_base.config._name_or_path)

txt_enc_base.to(DEVICE)
lbl_enc_base.to(DEVICE)
proj_base.to(DEVICE)

results = evaluate_model_with_label_names(
    txt_enc_base, lbl_enc_base, proj_base, 
    txt_tok, lbl_tok, 
    "Base"
)

# ==========================================================
# 4️⃣ RIEPILOGO FINALE
# ==========================================================
print("\n" + "="*60)
print("📊 RIEPILOGO FINALE")
print("="*60)
print(f"""
🏷️  Tipo embedding: LABEL NAMES (nomi brevi)
📏  Labels: {', '.join(label_names)}

🎯 Risultati:
   • Macro F1:  {results['macro_f1']:.4f}
   • Micro F1:  {results['micro_f1']:.4f}
   • Precision: {results['macro_precision']:.4f}
   • Recall:    {results['macro_recall']:.4f}
""")

print("="*60)
print("✅ Test completato!")
print("="*60)
