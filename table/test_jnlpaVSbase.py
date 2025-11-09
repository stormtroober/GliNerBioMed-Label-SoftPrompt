# -*- coding: utf-8 -*-
"""
Valutazione modello base vs fine-tunato su TEST SET token-level
====================================================================
Calcola Precision, Recall, F1 (macro e micro) confrontando:
- GLiNER-BioMed base (originale)
- GLiNER-BioMed fine-tunato (dai savings/)

Test set: dataset/test_dataset_tokenlevel.json
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
TEST_PATH = "dataset/test_dataset_tokenlevel.json"  # <- TEST SET (modificato)
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# 0️⃣ SEZIONE DI UTILITÀ
# ==========================================================
def select_checkpoint_interactive(savings_dir=SAVINGS_DIR):
    """Mostra menu interattivo per selezionare checkpoint."""
    checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        raise FileNotFoundError(f"❌ Nessun checkpoint trovato in {savings_dir}/")
    
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
    print("📦 SELEZIONE CHECKPOINT")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        date_str = datetime.datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {info['name']}")
        print(f"   📅 {date_str} | 💾 {info['size_mb']:.1f} MB")
    
    # Input utente
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
    
    # Se ci sono token speciali all'inizio/fine, preservali
    if tokens[0] == tokenizer.cls_token and tokens[-1] == tokenizer.sep_token and max_len >= 2:
        return [tokens[0]] + tokens[1:max_len-1] + [tokens[-1]]
    
    return tokens[:max_len]

# ==========================================================
# 1️⃣ CARICAMENTO DATI E CONFIGURAZIONE
# ==========================================================
print("📥 Caricamento configurazione...")
with open(LABEL2DESC_PATH) as f:
    label2desc = json.load(f)
with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())
num_labels = len(label_names)

print(f"\n✅ Caricate {num_labels} label:")
for i, name in enumerate(label_names):
    print(f"  [{i}] {name:15s}: {label2desc[name]}")

print(f"\n📚 Caricamento TEST SET da {TEST_PATH}...")  # <- Modificato
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

print(f"\n📦 Cartella savings: {SAVINGS_DIR}")
print(f"   Trovati {len([f for f in os.listdir(SAVINGS_DIR) if f.endswith('.pt')])} checkpoint")

# ==========================================================
# 2️⃣ FUNZIONE DI VALUTAZIONE PRINCIPALE
# ==========================================================
def evaluate_model_on_records(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name, records, dataset_type="test", save_results=True):
    """Valuta il modello su un set di record (test o training)."""
    print(f"\n🔍 Valutazione {model_name} su {len(records)} record ({dataset_type} set)...")
    
    txt_enc.eval()
    lbl_enc.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []
    n_skipped = 0
    
    with torch.no_grad():
        # Precomputa label embeddings (una sola volta)
        label_matrix = compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj, label_names).to(DEVICE)
        
        for idx, record in enumerate(records):
            # Estrai dati
            tokens = record["tokens"]
            labels = record["labels"]
            
            # Validazione lunghezza
            if len(tokens) != len(labels):
                print(f"⚠️  Record {idx}: mismatch lunghezza token-label ({len(tokens)} vs {len(labels)})")
                n_skipped += 1
                continue
            
            # Converti token in IDs e tronca se necessario
            input_ids = txt_tok.convert_tokens_to_ids(tokens)
            
            # Troncamento sicuro
            max_len = getattr(txt_tok, "model_max_length", 512)
            if len(input_ids) > max_len:
                print(f"⚠️  Record {idx}: troncato da {len(input_ids)} a {max_len} token")
                input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
                labels = labels[:len(input_ids)]
            
            # Crea tensore
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
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
    
    if n_skipped > 0:
        print(f"⚠️  Skipped {n_skipped} record con errori di formato")
    
    # Calcola metriche
    all_label_ids = list(range(num_labels))
    
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
        zero_division=0,
        digits=4
    )
    
    # Stampa risultati
    print(f"\n{'='*70}")
    print(f"📊 RISULTATI - {model_name} ({dataset_type.upper()} SET)")  # <- Modificato
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Totale record: {len(records)}")
    print(f"Record validi: {len(records) - n_skipped}")
    print(f"Token valutati: {len(y_true_all):,}")
    print(f"\n🎯 METRICHE AGGREGATE:")
    print(f"  Macro F1:       {f1_macro:.4f}")
    print(f"  Micro F1:       {f1_micro:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro):    {rec_macro:.4f}")
    print(f"  Precision (micro): {prec_micro:.4f}")
    print(f"  Recall (micro):    {rec_micro:.4f}")
    
    print(f"\n📋 REPORT PER CLASSE:")
    print(class_report)
    
    # Distribuzione predizioni vs ground truth
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
    
    # Salva risultati solo se richiesto
    if save_results:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/test_eval_{model_name.replace(' ', '_').replace('-', '_')}_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Risultati Test Set - {model_name}\n\n")
            f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {TEST_PATH}\n")
            f.write(f"**Record totali:** {len(records)}\n")
            f.write(f"**Record validi:** {len(records) - n_skipped}\n")
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
        
        print(f"💾 Risultati salvati in: {filename}")
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro,
        'n_tokens': len(y_true_all),
        'n_skipped': n_skipped
    }

# ==========================================================
# 3️⃣ MODELLO BASE
# ==========================================================
print("\n" + "="*70)
print("🔹 MODELLO BASE")
print("="*70)

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

results_base = evaluate_model_on_records(
    txt_enc_base, lbl_enc_base, proj_base, 
    txt_tok, lbl_tok, "Base", test_records, save_results=False
)

# ==========================================================
# 4️⃣ MODELLO FINE-TUNATO
# ==========================================================
print("\n" + "="*70)
print("🔸 MODELLO FINE-TUNATO")
print("="*70)

# Selezione checkpoint interattiva
checkpoint_path = select_checkpoint_interactive(SAVINGS_DIR)

# Carica checkpoint
print(f"\n📦 Caricamento checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
print(f"✅ Checkpoint caricato (epoch: {checkpoint.get('epoch', 'N/A')})")

# Costruisci modello fine-tunato
model_ft = GLiNER.from_pretrained(MODEL_NAME)
core_ft = model_ft.model

txt_enc_ft = core_ft.token_rep_layer.bert_layer.model
lbl_enc_ft = core_ft.token_rep_layer.labels_encoder.model
proj_ft = core_ft.token_rep_layer.labels_projection

# Applica i pesi salvati (controlla chiavi)
if 'soft_embeddings' in checkpoint:
    # Formato compatibile con training.py
    if hasattr(core_ft.token_rep_layer, 'soft_label_embeddings'):
        core_ft.token_rep_layer.soft_label_embeddings.load_state_dict(checkpoint['soft_embeddings'])
    proj_ft.load_state_dict(checkpoint['projection'])
    print("✅ Caricati soft embeddings + projection")
else:
    # Formato legacy
    lbl_enc_ft.load_state_dict(checkpoint.get('label_encoder_state_dict', {}))
    proj_ft.load_state_dict(checkpoint.get('projection_state_dict', {}))
    print("✅ Caricati state_dict legacy")

txt_enc_ft.to(DEVICE)
lbl_enc_ft.to(DEVICE)
proj_ft.to(DEVICE)

results_ft = evaluate_model_on_records(
    txt_enc_ft, lbl_enc_ft, proj_ft, 
    txt_tok, lbl_tok, "Fine-tuned", test_records, save_results=True
)

# ==========================================================
# 5️⃣ CONFRONTO FINALE
# ==========================================================
print("\n" + "="*70)
print("📊 CONFRONTO FINALE - TEST SET")  # <- Modificato
print("="*70)

print(f"\n{'Metrica':<20} {'Base':<12} {'Fine-tuned':<12} {'Δ':<10} {'% Migli.'}")
print("-" * 70)

metrics = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall']
metric_names = ['Macro F1', 'Micro F1', 'Macro Precision', 'Macro Recall']

improvements = []
for metric, name in zip(metrics, metric_names):
    base_val = results_base[metric]
    ft_val = results_ft[metric]
    delta = ft_val - base_val
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    
    if base_val > 0:
        perc_impr = (delta / base_val) * 100
        perc_str = f"{perc_impr:+.1f}%"
    else:
        perc_str = "N/A"
    
    print(f"{name:<20} {base_val:<12.4f} {ft_val:<12.4f} {delta_str:<10} {perc_str}")
    
    improvements.append(delta)

# Riassunto performance
avg_impr = sum(improvements) / len(improvements)
print(f"\n📈 Miglioramento medio su {len(improvements)} metriche: {avg_impr:+.4f}")
print(f"   (Compreso tra {min(improvements):+.4f} e {max(improvements):+.4f})")