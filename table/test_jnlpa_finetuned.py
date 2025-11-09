# -*- coding: utf-8 -*-
"""
Valutazione rapida modello fine-tunato su TEST SET
===================================================
Test secco senza confronto con il modello base.
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter
import os
import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "dataset/test_dataset_tokenlevel.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# UTILITÀ
# ==========================================================
def select_checkpoint_interactive(savings_dir=SAVINGS_DIR):
    """Mostra menu interattivo per selezionare checkpoint."""
    checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        raise FileNotFoundError(f"❌ Nessun checkpoint trovato in {savings_dir}/")
    
    checkpoints_info = []
    for ckpt in checkpoints:
        path = os.path.join(savings_dir, ckpt)
        mtime = os.path.getmtime(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        checkpoints_info.append({
            'name': ckpt,
            'path': path,
            'mtime': mtime,
            'size_mb': size_mb
        })
    
    checkpoints_info.sort(key=lambda x: x['mtime'], reverse=True)
    
    print("\n" + "="*60)
    print("📦 SELEZIONE CHECKPOINT")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        date_str = datetime.datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {info['name']}")
        print(f"   📅 {date_str} | 💾 {info['size_mb']:.1f} MB")
    
    while True:
        try:
            choice = input(f"\n👉 Seleziona checkpoint (1-{len(checkpoints_info)}) [default: 1]: ").strip()
            
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(checkpoints_info):
                selected = checkpoints_info[choice - 1]
                print(f"✅ Selezionato: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ Numero non valido. Scegli tra 1 e {len(checkpoints_info)}")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except KeyboardInterrupt:
            print("\n\n❌ Operazione annullata.")
            exit(0)

def compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj, label_names):
    """Embedda le descrizioni con encoder + projection."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)

def truncate_tokens_safe(tokens, tokenizer, max_len=None):
    """Tronca i token preservando [CLS]/[SEP] se presenti."""
    if max_len is None:
        max_len = tokenizer.model_max_length
    
    if len(tokens) <= max_len:
        return tokens
    
    if tokens[0] == tokenizer.cls_token and tokens[-1] == tokenizer.sep_token and max_len >= 2:
        return [tokens[0]] + tokens[1:max_len-1] + [tokens[-1]]
    
    return tokens[:max_len]

# ==========================================================
# CARICAMENTO DATI
# ==========================================================
print("📥 Caricamento configurazione...")
with open(LABEL2DESC_PATH) as f:
    label2desc = json.load(f)
with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())
num_labels = len(label_names)

print(f"✅ Caricate {num_labels} label")

print(f"\n📚 Caricamento TEST SET da {TEST_PATH}...")
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

# ==========================================================
# CARICAMENTO MODELLO FINE-TUNATO
# ==========================================================
print("\n" + "="*70)
print("🔸 CARICAMENTO MODELLO FINE-TUNATO")
print("="*70)

checkpoint_path = select_checkpoint_interactive(SAVINGS_DIR)

print(f"\n📦 Caricamento checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
print(f"✅ Checkpoint caricato (epoch: {checkpoint.get('epoch', 'N/A')})")

model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

# Applica i pesi salvati
if 'soft_embeddings' in checkpoint:
    if hasattr(core.token_rep_layer, 'soft_label_embeddings'):
        core.token_rep_layer.soft_label_embeddings.load_state_dict(checkpoint['soft_embeddings'])
    proj.load_state_dict(checkpoint['projection'])
    print("✅ Caricati soft embeddings + projection")
else:
    lbl_enc.load_state_dict(checkpoint.get('label_encoder_state_dict', {}))
    proj.load_state_dict(checkpoint.get('projection_state_dict', {}))
    print("✅ Caricati state_dict legacy")

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

txt_enc.to(DEVICE)
lbl_enc.to(DEVICE)
proj.to(DEVICE)

# ==========================================================
# VALUTAZIONE
# ==========================================================
print(f"\n🔍 Valutazione su {len(test_records)} record...")

txt_enc.eval()
lbl_enc.eval()
proj.eval()

y_true_all = []
y_pred_all = []
n_skipped = 0

with torch.no_grad():
    label_matrix = compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj, label_names).to(DEVICE)
    
    for idx, record in enumerate(test_records):
        tokens = record["tokens"]
        labels = record["labels"]
        
        if len(tokens) != len(labels):
            n_skipped += 1
            continue
        
        input_ids = txt_tok.convert_tokens_to_ids(tokens)
        
        max_len = getattr(txt_tok, "model_max_length", 512)
        if len(input_ids) > max_len:
            input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
            labels = labels[:len(input_ids)]
        
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)
        
        out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
        H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        logits = torch.matmul(H, label_matrix.T).squeeze(0)
        preds = logits.argmax(-1).cpu().numpy()
        
        for pred, true_label in zip(preds, labels):
            if true_label != -100:
                y_true_all.append(true_label)
                y_pred_all.append(pred)

# ==========================================================
# RISULTATI
# ==========================================================
all_label_ids = list(range(num_labels))

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="macro", zero_division=0, labels=all_label_ids
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="micro", zero_division=0, labels=all_label_ids
)

class_report = classification_report(
    y_true_all, y_pred_all, 
    target_names=label_names,
    labels=all_label_ids,
    zero_division=0,
    digits=4
)

print(f"\n{'='*70}")
print(f"📊 RISULTATI TEST SET")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
print(f"Record totali: {len(test_records)}")
print(f"Record validi: {len(test_records) - n_skipped}")
print(f"Token valutati: {len(y_true_all):,}")

print(f"\n🎯 METRICHE AGGREGATE:")
print(f"  Macro F1:          {f1_macro:.4f}")
print(f"  Micro F1:          {f1_micro:.4f}")
print(f"  Precision (macro): {prec_macro:.4f}")
print(f"  Recall (macro):    {rec_macro:.4f}")
print(f"  Precision (micro): {prec_micro:.4f}")
print(f"  Recall (micro):    {rec_micro:.4f}")

print(f"\n📋 REPORT PER CLASSE:")
print(class_report)

# Distribuzione
pred_counts = Counter(y_pred_all)
true_counts = Counter(y_true_all)

print(f"\n📈 DISTRIBUZIONE LABEL (Top 10):")
print(f"{'Label':<20} {'Pred':<10} {'True':<10} {'Diff':<10}")
print("-" * 50)
for label_id in sorted(pred_counts.keys(), key=lambda x: pred_counts[x], reverse=True)[:10]:
    label_name = id2label[label_id]
    pred_n = pred_counts[label_id]
    true_n = true_counts.get(label_id, 0)
    diff = pred_n - true_n
    diff_str = f"+{diff}" if diff >= 0 else str(diff)
    print(f"{label_name:<20} {pred_n:<10} {true_n:<10} {diff_str:<10}")

# Salvataggio risultati
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = "test_results"
os.makedirs(results_dir, exist_ok=True)

filename = f"{results_dir}/test_eval_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test - Modello Fine-tunato\n\n")
    f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Checkpoint:** {os.path.basename(checkpoint_path)}\n")
    f.write(f"**Epoca:** {checkpoint.get('epoch', 'N/A')}\n")
    f.write(f"**Dataset:** {TEST_PATH}\n")
    f.write(f"**Record totali:** {len(test_records)}\n")
    f.write(f"**Record validi:** {len(test_records) - n_skipped}\n")
    f.write(f"**Token valutati:** {len(y_true_all):,}\n\n")
    f.write(f"## Metriche aggregate\n\n")
    f.write(f"- **Macro F1:** {f1_macro:.4f}\n")
    f.write(f"- **Micro F1:** {f1_micro:.4f}\n")
    f.write(f"- **Precision (macro):** {prec_macro:.4f}\n")
    f.write(f"- **Recall (macro):** {rec_macro:.4f}\n")
    f.write(f"- **Precision (micro):** {prec_micro:.4f}\n")
    f.write(f"- **Recall (micro):** {rec_micro:.4f}\n\n")
    f.write(f"## Report per classe\n\n```\n{class_report}\n```\n")
    f.write(f"## Distribuzione label\n\n")
    f.write(f"| Label | Predette | Reali | Diff |\n")
    f.write(f"|-------|----------|-------|------|\n")
    for label_id in sorted(pred_counts.keys(), key=lambda x: pred_counts[x], reverse=True):
        label_name = id2label[label_id]
        pred_n = pred_counts[label_id]
        true_n = true_counts.get(label_id, 0)
        diff = pred_n - true_n
        f.write(f"| {label_name} | {pred_n} | {true_n} | {diff:+d} |\n")

print(f"\n💾 Risultati salvati in: {filename}")
print(f"\n✅ Test completato!")