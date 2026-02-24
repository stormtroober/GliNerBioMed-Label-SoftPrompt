# -*- coding: utf-8 -*-
"""
Valutazione modello base vs fine-tunato su test set token-level
================================================================
Calcola Precision, Recall, F1 (macro e micro) confrontando:
- GLiNER-BioMed base (originale)
- GLiNER-BioMed fine-tunato (dai savings/)

Test set: test_dataset_tokenlevel.json
"""

import json
from numpy import record
import torch
import torch.nn.functional as F
from gliner import GLiNER
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter, defaultdict
import numpy as np
import os
import time
from datetime import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
path = "../dataset_bc5cdr"
TEST_PATH = path + "/test_dataset_tknlvl_bi.json"
TEST_SPAN_PATH = path + "/test_dataset_span_bi.json"
LABEL2DESC_PATH = path + "/label2desc.json"
LABEL2ID_PATH = path + "/label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

# Trova ultimo checkpoint salvato
savings_dir = "savings"
checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
if not checkpoints:
    raise FileNotFoundError("Nessun checkpoint trovato in savings/")
latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(savings_dir, x)))
CHECKPOINT_PATH = os.path.join(savings_dir, latest_checkpoint)

print(f"📦 Checkpoint selezionato: {CHECKPOINT_PATH}")

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

print(f"📚 Caricamento test set (token-level) da {TEST_PATH}...")
with open(TEST_PATH, "r") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test (token-level)")

print(f"📚 Caricamento test set (span-level) da {TEST_SPAN_PATH}...")
with open(TEST_SPAN_PATH, "r") as f:
    span_records = json.load(f)
print(f"✅ Caricati {len(span_records)} esempi di test (span-level)")

if len(test_records) != len(span_records):
    print(f"⚠️  ATTENZIONE: numero di esempi diverso ({len(test_records)} vs {len(span_records)}). "
          "La valutazione span potrebbe essere inaccurata.")

# ==========================================================
# 2️⃣ FUNZIONI HELPER
# ==========================================================
def select_checkpoint_interactive(savings_dir="savings"):
    """Mostra menu interattivo per selezionare checkpoint."""
    checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        raise FileNotFoundError(f"Nessun checkpoint trovato in {savings_dir}/")
    
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
        import datetime
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


def compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj):
    """Embedda le descrizioni con encoder + projection."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)


def get_word_boundary_flags(subword_tokens):
    """
    Dato un elenco di subword token, restituisce una lista di booleani:
    True se quel token è il PRIMO subword di una parola word-level, False altrimenti.
    I token speciali ([CLS]/[SEP] ecc.) vengono marcati come False.
    """
    flags = []
    for tok in subword_tokens:
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            flags.append(False)
        elif tok.startswith("▁") or tok.startswith("Ġ"):
            flags.append(True)
        elif not flags:
            flags.append(True)
        else:
            flags.append(False)
    return flags


def subword_preds_to_word_preds(subword_tokens, subword_preds, subword_labels):
    """
    Mappa le predizioni subword → predizioni word-level usando la strategia
    'first-subword': la label del primo subword di ogni parola è la label della parola.
    """
    flags = get_word_boundary_flags(subword_tokens)
    word_preds, word_trues = [], []
    for flag, pred, true in zip(flags, subword_preds, subword_labels):
        if flag and true != -100:
            word_preds.append(pred)
            word_trues.append(true)
    return word_preds, word_trues


def extract_spans_from_word_seq(word_labels, background_label_id):
    """
    Dato un elenco di label word-level, estrae span contigui dello stesso label
    (diverso da background_label_id) come set di tuple (start, end, label_id).
    """
    spans = set()
    n = len(word_labels)
    i = 0
    while i < n:
        lbl = word_labels[i]
        if lbl != background_label_id:
            j = i
            while j < n and word_labels[j] == lbl:
                j += 1
            spans.add((i, j - 1, lbl))
            i = j
        else:
            i += 1
    return spans


def compute_span_metrics(tp_dict, fp_dict, fn_dict, support_dict, id2label, background_label_id):
    """
    Calcola metriche span-based per label e aggregate (macro/micro).
    """
    all_ids = sorted(set(list(tp_dict.keys()) + list(fp_dict.keys()) + list(fn_dict.keys())))
    all_ids = [lid for lid in all_ids if lid != background_label_id]

    p_list, r_list, f1_list = [], [], []
    report_lines = []
    header = f"{'Label':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>10}"
    report_lines.append(header)
    report_lines.append("-" * 65)

    for lid in all_ids:
        tp = tp_dict[lid]; fp = fp_dict[lid]; fn = fn_dict[lid]; sup = support_dict[lid]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        name = id2label.get(lid, str(lid))
        p_list.append(p); r_list.append(r); f1_list.append(f1)
        report_lines.append(f"{name:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {sup:>10}")

    macro_p  = float(np.mean(p_list))  if p_list  else 0.0
    macro_r  = float(np.mean(r_list))  if r_list  else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    total_tp = sum(tp_dict[l] for l in all_ids)
    total_fp = sum(fp_dict[l] for l in all_ids)
    total_fn = sum(fn_dict[l] for l in all_ids)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, report_lines


def evaluate_model(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name, checkpoint_info=None, span_records=None):
    """Valuta il modello sul test set (token-level + opzionale span-level)."""
    print(f"\n🔍 Valutazione {model_name}...")
    
    txt_enc.eval()
    lbl_enc.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []

    # Identifica classe 'O' e costruisce subset di label rilevanti (come test_mono.py)
    background_id = label2id.get("O", 5)
    ignore_index = -1
    for _idx, _name in enumerate(label_names):
        if _name == 'O':
            ignore_index = _idx
            break
    all_label_ids = list(range(len(label_names)))
    if ignore_index != -1:
        relevant_label_ids  = [i for i in all_label_ids if i != ignore_index]
        relevant_label_names = [label_names[i] for i in relevant_label_ids]
        print(f"ℹ️  Esclusione classe 'O' (ID: {ignore_index}) dalle metriche token-level.")
    else:
        relevant_label_ids  = all_label_ids
        relevant_label_names = label_names

    # Span-based accumulatori
    span_tp = defaultdict(int)
    span_fp = defaultdict(int)
    span_fn = defaultdict(int)
    span_support = defaultdict(int)
    n_span_skipped = 0

    total_records = len(test_records)
    checkpoint_interval = max(1, total_records // 5)
    
    with torch.no_grad():
        label_matrix = compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj).to(DEVICE)
        
        infer_start_time = time.time()
        for idx, record in enumerate(test_records):
            tokens = record["tokens"]
            labels = record["labels"]

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
            labels = record["labels"]
            
            # Forward pass
            out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            logits = torch.matmul(H, label_matrix.T).squeeze(0)
            preds = logits.argmax(-1).cpu().numpy()
            subword_preds_list = preds.tolist()
            
            # Token-level metrics
            for pred, true_label in zip(subword_preds_list, labels):
                if true_label != -100:
                    y_true_all.append(true_label)
                    y_pred_all.append(pred)

            # ---- Span-based evaluation ----
            if span_records is not None:
                if idx < len(span_records):
                    span_rec = span_records[idx]
                    gt_ner = span_rec.get("ner", [])

                    gt_spans = set()
                    for s, e, lbl_str in gt_ner:
                        lbl_int = int(lbl_str)
                        if lbl_int != background_id:
                            gt_spans.add((s, e, lbl_int))
                            span_support[lbl_int] += 1

                    word_preds, _ = subword_preds_to_word_preds(tokens, subword_preds_list, labels)
                    pred_spans = extract_spans_from_word_seq(word_preds, background_id)

                    for span in pred_spans:
                        if span in gt_spans:
                            span_tp[span[2]] += 1
                        else:
                            span_fp[span[2]] += 1
                    for span in gt_spans:
                        if span not in pred_spans:
                            span_fn[span[2]] += 1
                else:
                    n_span_skipped += 1

            # Progress checkpoint ogni ~20% (come test_mono.py)
            loop_idx = idx + 1
            if loop_idx % checkpoint_interval == 0 or loop_idx == total_records:
                if len(y_true_all) > 0:
                    progress = (loop_idx / total_records) * 100
                    _mf1 = precision_recall_fscore_support(
                        y_true_all, y_pred_all, labels=relevant_label_ids,
                        average="macro", zero_division=0
                    )[2]
                    _uf1 = precision_recall_fscore_support(
                        y_true_all, y_pred_all, labels=relevant_label_ids,
                        average="micro", zero_division=0
                    )[2]
                    print(f"\n [{progress:5.1f}%] Macro F1: {_mf1:.4f} | Micro F1: {_uf1:.4f} | Tokens: {len(y_true_all):,}")

    infer_time = time.time() - infer_start_time
    samples_per_sec = len(test_records) / infer_time if infer_time > 0 else 0
    tokens_per_sec = len(y_true_all) / infer_time if infer_time > 0 else 0

    if span_records is not None and n_span_skipped > 0:
        print(f"⚠️  {n_span_skipped} esempi saltati nella valutazione span (indice fuori range).")

    # Calcola metriche token-level (esclusa 'O' come in test_mono.py)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0, labels=relevant_label_ids
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="micro", zero_division=0, labels=relevant_label_ids
    )
    
    class_report = classification_report(
        y_true_all, y_pred_all, 
        target_names=relevant_label_names,
        labels=relevant_label_ids,
        zero_division=0
    )
    
    # Stampa risultati token-level
    print(f"\n{'='*60}")
    print(f"📊 RISULTATI - {model_name}")
    print(f"{'='*60}")
    print(f"Totale token valutati: {len(y_true_all)}")
    print(f"Tempo di inferenza:  {infer_time:.2f} s")
    print(f"Velocità:            {samples_per_sec:.2f} samples/s | {tokens_per_sec:.2f} tokens/s")
    print(f"\n🎯 METRICHE TOKEN-LEVEL AGGREGATE (No 'O' class):")
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

    # Calcola e stampa metriche span-based
    span_macro_p = span_macro_r = span_macro_f1 = 0.0
    span_micro_p = span_micro_r = span_micro_f1 = 0.0
    span_report_lines = []

    if span_records is not None:
        span_macro_p, span_macro_r, span_macro_f1, \
        span_micro_p, span_micro_r, span_micro_f1, \
        span_report_lines = compute_span_metrics(
            span_tp, span_fp, span_fn, span_support, id2label, background_id
        )
        print(f"\n{'='*60}")
        print(f"🎯 METRICHE SPAN-BASED (Exact Match, escluso 'O')")
        print(f"{'='*60}")
        print(f"  Macro P: {span_macro_p:.4f} | Macro R: {span_macro_r:.4f} | Macro F1: {span_macro_f1:.4f}")
        print(f"  Micro P: {span_micro_p:.4f} | Micro R: {span_micro_r:.4f} | Micro F1: {span_micro_f1:.4f}")
        print(f"\n📋 REPORT SPAN PER CLASSE:")
        for line in span_report_lines:
            print(line)
    
    # Salva risultati
    os.makedirs("testresults", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{model_name.replace(' ', '_').replace('-', '_')}_{timestamp}.md"
    
    with open(f'testresults/{filename}', "w") as f:
        f.write(f"# Risultati - {model_name}\n\n")
        f.write(f"**Data test:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if checkpoint_info:
            f.write(f"## Informazioni Checkpoint\n\n")
            f.write(f"**Checkpoint path:** `{checkpoint_info.get('checkpoint_path', 'N/A')}`\n\n")
            
            f.write(f"### Dataset di Training\n\n")
            f.write(f"- **Nome dataset:** {checkpoint_info.get('dataset_name', 'N/A')}\n")
            f.write(f"- **Path dataset:** `{checkpoint_info.get('dataset_path', 'N/A')}`\n")
            f.write(f"- **Dimensione dataset:** {checkpoint_info.get('dataset_size', 'N/A')} samples\n\n")
            
            hyperparams = checkpoint_info.get('hyperparameters', {})
            if hyperparams:
                f.write(f"### Iperparametri di Training\n\n")
                f.write(f"| Parametro | Valore |\n")
                f.write(f"|-----------|--------|\n")
                f.write(f"| Batch Size | {hyperparams.get('batch_size', 'N/A')} |\n")
                f.write(f"| Epochs | {hyperparams.get('epochs', 'N/A')} |\n")
                f.write(f"| Learning Rate | {hyperparams.get('learning_rate', 'N/A')} |\n")
                f.write(f"| Weight Decay | {hyperparams.get('weight_decay', 'N/A')} |\n")
                f.write(f"| Temperature | {hyperparams.get('temperature', 'N/A')} |\n")
                f.write(f"| Gradient Clip | {hyperparams.get('grad_clip', 'N/A')} |\n")
                f.write(f"| Warmup Steps | {hyperparams.get('warmup_steps', 'N/A')} |\n")
                f.write(f"| Early Stopping Patience | {hyperparams.get('early_stopping_patience', 'N/A')} |\n")
                f.write(f"| Random Seed | {hyperparams.get('random_seed', 'N/A')} |\n\n")
            
            training_info = checkpoint_info.get('training_info', {})
            if training_info:
                f.write(f"### Risultati Training\n\n")
                f.write(f"- **Best Loss:** {training_info.get('best_loss', 'N/A'):.4f}\n")
                f.write(f"- **Best Epoch:** {training_info.get('best_epoch', 'N/A')}\n")
                training_time = training_info.get('total_training_time_seconds', 0)
                f.write(f"- **Tempo Training:** {training_time:.1f}s ({training_time/60:.1f} min)\n")
                f.write(f"- **Modello Base:** `{training_info.get('model_name', 'N/A')}`\n\n")
        
        f.write(f"## Risultati Test\n\n")
        f.write(f"- **Token valutati:** {len(y_true_all)}\n")
        f.write(f"- **Tempo Inferenza:** {infer_time:.2f} s\n")
        f.write(f"- **Velocità:** {samples_per_sec:.2f} samples/s | {tokens_per_sec:.2f} tokens/s\n\n")
        
        f.write(f"### Metriche Token-Level Aggregate (No 'O' class)\n\n")
        f.write(f"| Metrica | Valore |\n")
        f.write(f"|---------|--------|\n")
        f.write(f"| Macro F1 | {f1_macro:.4f} |\n")
        f.write(f"| Micro F1 | {f1_micro:.4f} |\n")
        f.write(f"| Precision (macro) | {prec_macro:.4f} |\n")
        f.write(f"| Recall (macro) | {rec_macro:.4f} |\n")
        f.write(f"| Precision (micro) | {prec_micro:.4f} |\n")
        f.write(f"| Recall (micro) | {rec_micro:.4f} |\n\n")
        
        f.write(f"### Distribuzione Predizioni\n\n")
        f.write(f"| Label | Predizioni | Ground Truth |\n")
        f.write(f"|-------|------------|-------------|\n")
        for label_id in sorted(true_counts.keys(), key=lambda x: true_counts[x], reverse=True):
            label_name = id2label[label_id]
            f.write(f"| {label_name} | {pred_counts.get(label_id, 0)} | {true_counts[label_id]} |\n")
        f.write(f"\n")
        
        f.write(f"### Report Token-Level per Classe\n\n```\n{class_report}\n```\n")

        if span_records is not None:
            f.write(f"\n## Metriche Span-Based (Exact Match, escluso 'O')\n")
            f.write(f"| Metric | Precision | Recall | F1 |\n")
            f.write(f"|--------|-----------|--------|---------|\n")
            f.write(f"| **Macro** | {span_macro_p:.4f} | {span_macro_r:.4f} | **{span_macro_f1:.4f}** |\n")
            f.write(f"| **Micro** | {span_micro_p:.4f} | {span_micro_r:.4f} | **{span_micro_f1:.4f}** |\n")
            f.write(f"\n### Report Span per Classe\n```\n")
            for line in span_report_lines:
                f.write(line + "\n")
            f.write("```\n")
    
    print(f"\n💾 Risultati salvati in: testresults/{filename}")
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro,
        'span_macro_f1': span_macro_f1,
        'span_micro_f1': span_micro_f1,
    }

# ==========================================================
# 3️⃣ VALUTAZIONE MODELLO FINE-TUNATO
# ==========================================================
print("\n" + "="*60)
print("🔸 MODELLO FINE-TUNATO")
print("="*60)

CHECKPOINT_PATH = select_checkpoint_interactive("savings")

model_ft = GLiNER.from_pretrained(MODEL_NAME)
core_ft = model_ft.model

txt_enc_ft = core_ft.token_rep_layer.bert_layer.model
lbl_enc_ft = core_ft.token_rep_layer.labels_encoder.model
proj_ft = core_ft.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc_ft.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc_ft.config._name_or_path)

print(f"📦 Caricamento checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

checkpoint_info = {
    'checkpoint_path': CHECKPOINT_PATH,
    'dataset_name': checkpoint.get('dataset_name', 'N/A'),
    'dataset_path': checkpoint.get('dataset_path', 'N/A'),
    'dataset_size': checkpoint.get('dataset_size', 'N/A'),
    'hyperparameters': checkpoint.get('hyperparameters', {}),
    'training_info': checkpoint.get('training_info', {}),
}

print(f"📊 Dataset di training: {checkpoint_info['dataset_name']} ({checkpoint_info['dataset_size']} samples)")
if checkpoint_info['hyperparameters']:
    hp = checkpoint_info['hyperparameters']
    print(f"⚙️  Hyperparameters: bs={hp.get('batch_size')}, lr={hp.get('learning_rate')}, epochs={hp.get('epochs')}")

lbl_enc_ft.load_state_dict(checkpoint['label_encoder_state_dict'])
proj_ft.load_state_dict(checkpoint['projection_state_dict'])

txt_enc_ft.to(DEVICE)
lbl_enc_ft.to(DEVICE)
proj_ft.to(DEVICE)

results_ft = evaluate_model(
    txt_enc_ft, lbl_enc_ft, proj_ft, txt_tok, lbl_tok,
    "Fine_tuned", checkpoint_info, span_records=span_records
)

print("\n✅ Valutazione completata!")