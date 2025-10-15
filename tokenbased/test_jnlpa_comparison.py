# -*- coding: utf-8 -*-
"""
Confronto tra due checkpoint fine-tunati
==========================================
Permette di selezionare e confrontare due diversi checkpoint salvati.

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
# 1️⃣ CARICAMENTO LABELS E TEST SET
# ==========================================================
print("📥 Caricamento configurazione...")
with open(LABEL2DESC_PATH) as f:
    label2desc = json.load(f)
with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())

print(f"📚 Caricamento test set da {TEST_PATH}...")
with open(TEST_PATH, "r") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

# ==========================================================
# 2️⃣ FUNZIONI HELPER
# ==========================================================
def select_checkpoint_interactive(savings_dir="savings", prompt_text="checkpoint"):
    """Mostra menu interattivo per selezionare checkpoint."""
    checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        raise FileNotFoundError(f"Nessun checkpoint trovato in {savings_dir}/")
    
    # Ordina per data di modifica (più recente prima)
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
    
    # Mostra menu
    print("\n" + "="*60)
    print(f"📦 SELEZIONE {prompt_text.upper()}")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        import datetime
        date_str = datetime.datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {info['name']}")
        print(f"   📅 {date_str} | 💾 {info['size_mb']:.1f} MB")
    
    # Input utente
    while True:
        try:
            choice = input(f"\n👉 Seleziona {prompt_text} (1-{len(checkpoints_info)}) [default: 1]: ").strip()
            
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(checkpoints_info):
                selected = checkpoints_info[choice - 1]
                print(f"✅ Selezionato: {selected['name']}")
                return selected['path'], selected['name']
            else:
                print(f"❌ Numero non valido. Scegli tra 1 e {len(checkpoints_info)}")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except KeyboardInterrupt:
            print("\n\n❌ Operazione annullata.")
            exit(0)

def compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj):
    """Embedda le descrizioni con encoder + projection."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)

def evaluate_model(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name):
    """Valuta il modello sul test set."""
    print(f"\n🔍 Valutazione {model_name}...")
    
    txt_enc.eval()
    lbl_enc.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        # Precomputa label embeddings
        label_matrix = compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj).to(DEVICE)
        
        for record in test_records:
            labels = record["labels"]
            tokens = record["tokens"]

            ids = txt_tok.convert_tokens_to_ids(tokens)

            # Troncamento sicuro
            max_len = getattr(txt_tok, "model_max_length", 0) or 0
            if max_len > 0 and len(ids) > max_len:
                if len(ids) >= 2 and max_len >= 2:
                    middle_budget = max_len - 2
                    ids = [ids[0]] + ids[1:1+middle_budget] + [ids[-1]]
                else:
                    ids = ids[:max_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)
            
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
    print(f"📊 RISULTATI - {model_name}")
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
    
    print(f"\n📈 DISTRIBUZIONE (Top 5):")
    print(f"{'Label':<15} {'Pred':<8} {'True':<8}")
    print("-" * 35)
    for label_id in sorted(pred_counts.keys(), key=lambda x: pred_counts[x], reverse=True)[:5]:
        label_name = id2label[label_id]
        print(f"{label_name:<15} {pred_counts[label_id]:<8} {true_counts.get(label_id, 0):<8}")
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro,
        'total_tokens': len(y_true_all)
    }

def load_checkpoint_model(checkpoint_path, model_name):
    """Carica un modello con il checkpoint specificato."""
    model = GLiNER.from_pretrained(MODEL_NAME)
    core = model.model
    
    txt_enc = core.token_rep_layer.bert_layer.model
    lbl_enc = core.token_rep_layer.labels_encoder.model
    proj = core.token_rep_layer.labels_projection
    
    # Carica checkpoint
    print(f"📦 Caricamento checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    lbl_enc.load_state_dict(checkpoint['label_encoder_state_dict'])
    proj.load_state_dict(checkpoint['projection_state_dict'])
    
    txt_enc.to(DEVICE)
    lbl_enc.to(DEVICE)
    proj.to(DEVICE)
    
    return txt_enc, lbl_enc, proj

# ==========================================================
# 3️⃣ SELEZIONE CHECKPOINT
# ==========================================================
print("\n" + "="*60)
print("🔸 CONFRONTO TRA DUE CHECKPOINT")
print("="*60)

# Selezione primo checkpoint
checkpoint1_path, checkpoint1_name = select_checkpoint_interactive(SAVINGS_DIR, "primo checkpoint")

# Selezione secondo checkpoint
checkpoint2_path, checkpoint2_name = select_checkpoint_interactive(SAVINGS_DIR, "secondo checkpoint")

# Verifica che non siano lo stesso
if checkpoint1_path == checkpoint2_path:
    print("\n⚠️  ATTENZIONE: Hai selezionato lo stesso checkpoint due volte!")
    print("Il confronto continuerà comunque...\n")

# ==========================================================
# 4️⃣ VALUTAZIONE CHECKPOINT 1
# ==========================================================
print("\n" + "="*60)
print(f"🔹 CHECKPOINT 1: {checkpoint1_name}")
print("="*60)

txt_enc1, lbl_enc1, proj1 = load_checkpoint_model(checkpoint1_path, checkpoint1_name)
txt_tok = AutoTokenizer.from_pretrained(txt_enc1.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc1.config._name_or_path)

results1 = evaluate_model(txt_enc1, lbl_enc1, proj1, txt_tok, lbl_tok, f"Checkpoint 1")

# Libera memoria
del txt_enc1, lbl_enc1, proj1
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ==========================================================
# 5️⃣ VALUTAZIONE CHECKPOINT 2
# ==========================================================
print("\n" + "="*60)
print(f"🔸 CHECKPOINT 2: {checkpoint2_name}")
print("="*60)

txt_enc2, lbl_enc2, proj2 = load_checkpoint_model(checkpoint2_path, checkpoint2_name)

results2 = evaluate_model(txt_enc2, lbl_enc2, proj2, txt_tok, lbl_tok, f"Checkpoint 2")

# ==========================================================
# 6️⃣ CONFRONTO FINALE
# ==========================================================
print("\n" + "="*60)
print("📊 CONFRONTO FINALE")
print("="*60)
print(f"\nCheckpoint 1: {checkpoint1_name}")
print(f"Checkpoint 2: {checkpoint2_name}")
print("\n" + "-"*60)
print(f"\n{'Metrica':<20} {'Ckpt 1':<12} {'Ckpt 2':<12} {'Δ':<10}")
print("-" * 60)

metrics = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 'micro_precision', 'micro_recall']
metric_names = ['Macro F1', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Micro Precision', 'Micro Recall']

for metric, name in zip(metrics, metric_names):
    val1 = results1[metric]
    val2 = results2[metric]
    delta = val2 - val1
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    
    # Evidenzia il migliore
    marker1 = "🏆 " if val1 > val2 else "   "
    marker2 = "🏆 " if val2 > val1 else "   "
    
    print(f"{name:<20} {marker1}{val1:<9.4f} {marker2}{val2:<9.4f} {delta_str:<10}")

print("\n" + "="*60)
print("✅ Confronto completato!")
print("="*60)

# Riepilogo miglioramenti
improvements = sum(1 for m in metrics if results2[m] > results1[m])
print(f"\n📈 Checkpoint 2 migliore in {improvements}/{len(metrics)} metriche")