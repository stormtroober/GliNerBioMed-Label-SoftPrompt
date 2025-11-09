# -*- coding: utf-8 -*-
"""
Confronto tra due modelli fine-tunati su TEST SET token-level
====================================================================
Calcola Precision, Recall, F1 (macro e micro) confrontando:
- GLiNER-BioMed fine-tunato #1 (dai savings/)
- GLiNER-BioMed fine-tunato #2 (dai savings/)

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
TEST_PATH = "dataset/test_dataset_tokenlevel.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# 0️⃣ SEZIONE DI UTILITÀ
# ==========================================================
def select_checkpoint_interactive(savings_dir=SAVINGS_DIR, model_label="Modello"):
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
    print(f"📦 SELEZIONE CHECKPOINT - {model_label}")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        date_str = datetime.datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {info['name']}")
        print(f"   📅 {date_str} | 💾 {info['size_mb']:.1f} MB")
    
    # Input utente
    while True:
        try:
            choice = input(f"\n👉 Seleziona checkpoint per {model_label} (1-{len(checkpoints_info)}) [default: 1]: ").strip()
            
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

print(f"\n📚 Caricamento TEST SET da {TEST_PATH}...")
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

print(f"\n📦 Cartella savings: {SAVINGS_DIR}")
print(f"   Trovati {len([f for f in os.listdir(SAVINGS_DIR) if f.endswith('.pt')])} checkpoint")

# ==========================================================
# 2️⃣ FUNZIONE DI VALUTAZIONE PRINCIPALE
# ==========================================================
def evaluate_model_on_records(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name, records):
    """Valuta il modello su un set di record."""
    print(f"\n🔍 Valutazione {model_name} su {len(records)} record...")
    
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
    print(f"📊 RISULTATI - {model_name}")
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
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro,
        'n_tokens': len(y_true_all),
        'n_skipped': n_skipped,
        'class_report': class_report,
        'pred_counts': pred_counts,
        'true_counts': true_counts
    }

def load_finetuned_model(checkpoint_path, model_name_short):
    """Carica un modello fine-tunato da checkpoint."""
    print(f"\n📦 Caricamento checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"✅ Checkpoint caricato (epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # Costruisci modello
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
    
    txt_enc.to(DEVICE)
    lbl_enc.to(DEVICE)
    proj.to(DEVICE)
    
    return txt_enc, lbl_enc, proj

# ==========================================================
# 3️⃣ SELEZIONE E CARICAMENTO MODELLI
# ==========================================================
print("\n" + "="*70)
print("🔸 SELEZIONE MODELLI FINE-TUNATI")
print("="*70)

# Selezione primo modello
checkpoint1_path, checkpoint1_name = select_checkpoint_interactive(SAVINGS_DIR, "Modello 1")
txt_enc1, lbl_enc1, proj1 = load_finetuned_model(checkpoint1_path, "Model-1")

# Selezione secondo modello
print("\n")
checkpoint2_path, checkpoint2_name = select_checkpoint_interactive(SAVINGS_DIR, "Modello 2")

# Verifica che non sia lo stesso checkpoint
if checkpoint1_path == checkpoint2_path:
    print("\n⚠️  ATTENZIONE: Hai selezionato lo stesso checkpoint due volte!")
    response = input("Vuoi continuare comunque? (s/n) [n]: ").strip().lower()
    if response != 's':
        print("❌ Operazione annullata.")
        exit(0)

txt_enc2, lbl_enc2, proj2 = load_finetuned_model(checkpoint2_path, "Model-2")

# Carica tokenizer (condivisi tra i modelli)
txt_tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
lbl_tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

# ==========================================================
# 4️⃣ VALUTAZIONE MODELLI
# ==========================================================
print("\n" + "="*70)
print("🔹 VALUTAZIONE MODELLO 1")
print("="*70)
results1 = evaluate_model_on_records(
    txt_enc1, lbl_enc1, proj1, 
    txt_tok, lbl_tok, checkpoint1_name, test_records
)

print("\n" + "="*70)
print("🔸 VALUTAZIONE MODELLO 2")
print("="*70)
results2 = evaluate_model_on_records(
    txt_enc2, lbl_enc2, proj2, 
    txt_tok, lbl_tok, checkpoint2_name, test_records
)

# ==========================================================
# 5️⃣ CONFRONTO FINALE
# ==========================================================
print("\n" + "="*70)
print("📊 CONFRONTO FINALE - DUE MODELLI FINE-TUNATI")
print("="*70)

print(f"\n🔹 Modello 1: {checkpoint1_name}")
print(f"🔸 Modello 2: {checkpoint2_name}")

print(f"\n{'Metrica':<20} {'Modello 1':<12} {'Modello 2':<12} {'Δ':<10} {'Migliore'}")
print("-" * 75)

metrics = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 'micro_precision', 'micro_recall']
metric_names = ['Macro F1', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Micro Precision', 'Micro Recall']

improvements = []
wins_model1 = 0
wins_model2 = 0

for metric, name in zip(metrics, metric_names):
    val1 = results1[metric]
    val2 = results2[metric]
    delta = val2 - val1
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    
    if val1 > val2:
        winner = "🔹 Modello 1"
        wins_model1 += 1
    elif val2 > val1:
        winner = "🔸 Modello 2"
        wins_model2 += 1
    else:
        winner = "⚖️  Pari"
    
    print(f"{name:<20} {val1:<12.4f} {val2:<12.4f} {delta_str:<10} {winner}")
    improvements.append(delta)

# Riassunto performance
print(f"\n{'='*75}")
print(f"📈 RIASSUNTO:")
print(f"   🔹 Modello 1 vince su: {wins_model1}/{len(metrics)} metriche")
print(f"   🔸 Modello 2 vince su: {wins_model2}/{len(metrics)} metriche")

if wins_model1 > wins_model2:
    print(f"\n🏆 VINCITORE: Modello 1 ({checkpoint1_name})")
elif wins_model2 > wins_model1:
    print(f"\n🏆 VINCITORE: Modello 2 ({checkpoint2_name})")
else:
    print(f"\n⚖️  RISULTATO: Pareggio")

avg_delta = sum(improvements) / len(improvements)
print(f"\n📊 Delta medio (Modello 2 - Modello 1): {avg_delta:+.4f}")
print(f"   (Range: {min(improvements):+.4f} ÷ {max(improvements):+.4f})")

# ==========================================================
# 6️⃣ SALVATAGGIO RISULTATI
# ==========================================================
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = "test_results"
os.makedirs(results_dir, exist_ok=True)

filename = f"{results_dir}/comparison_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Confronto tra due modelli fine-tunati\n\n")
    f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Dataset:** {TEST_PATH}\n")
    f.write(f"**Record totali:** {len(test_records)}\n\n")
    
    f.write(f"## Modelli confrontati\n\n")
    f.write(f"1. **Modello 1:** `{checkpoint1_name}`\n")
    f.write(f"   - Path: `{checkpoint1_path}`\n\n")
    f.write(f"2. **Modello 2:** `{checkpoint2_name}`\n")
    f.write(f"   - Path: `{checkpoint2_path}`\n\n")
    
    f.write(f"## Confronto metriche\n\n")
    f.write(f"| Metrica | Modello 1 | Modello 2 | Δ | Migliore |\n")
    f.write(f"|---------|-----------|-----------|---|----------|\n")
    
    for metric, name in zip(metrics, metric_names):
        val1 = results1[metric]
        val2 = results2[metric]
        delta = val2 - val1
        winner = "Modello 1" if val1 > val2 else ("Modello 2" if val2 > val1 else "Pari")
        f.write(f"| {name} | {val1:.4f} | {val2:.4f} | {delta:+.4f} | {winner} |\n")
    
    f.write(f"\n## Riassunto\n\n")
    f.write(f"- **Modello 1 vince su:** {wins_model1}/{len(metrics)} metriche\n")
    f.write(f"- **Modello 2 vince su:** {wins_model2}/{len(metrics)} metriche\n")
    f.write(f"- **Delta medio:** {avg_delta:+.4f}\n\n")
    
    if wins_model1 > wins_model2:
        f.write(f"**🏆 VINCITORE:** Modello 1 ({checkpoint1_name})\n\n")
    elif wins_model2 > wins_model1:
        f.write(f"**🏆 VINCITORE:** Modello 2 ({checkpoint2_name})\n\n")
    else:
        f.write(f"**⚖️ RISULTATO:** Pareggio\n\n")
    
    f.write(f"## Report dettagliato Modello 1\n\n```\n{results1['class_report']}\n```\n\n")
    f.write(f"## Report dettagliato Modello 2\n\n```\n{results2['class_report']}\n```\n")

print(f"\n💾 Risultati salvati in: {filename}")
print(f"\n✅ Confronto completato!")