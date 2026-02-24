# -*- coding: utf-8 -*-
"""
Valutazione modello fine-tunato con Soft Prompting su TEST SET.
Versione SOFT TABLE Only.
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
import argparse
import re
import time

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH_DATASET = "../dataset"
#PATH_DATASET = "../dataset_bc5cdr"

TEST_PATH = PATH_DATASET + "/test_dataset_tknlvl_bi.json"
TEST_SPAN_PATH = PATH_DATASET + "/test_dataset_span_bi.json"
LABEL2DESC_PATH = PATH_DATASET + "/label2desc.json"
LABEL2ID_PATH = PATH_DATASET + "/label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# UTILITÀ
# ==========================================================
def select_checkpoint_interactive(savings_dir=SAVINGS_DIR):
    """Mostra menu interattivo per selezionare checkpoint."""
    if not os.path.exists(savings_dir):
        print(f"Directory {savings_dir} non trovata.")
        return None

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

def compute_label_matrix_from_soft_table(soft_table, proj):
    """Usa i vettori ottimizzati nella Soft Table (Modalità SOFT)."""
    # soft_table.weight è [Num_Labels, Hidden_Dim]
    vecs = proj(soft_table.weight)
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
# SPAN-BASED EVALUATION UTILITIES
# ==========================================================

def get_word_boundary_flags(subword_tokens):
    """
    Dato un elenco di subword token (SentencePiece con prefisso '▁' o token speciali
    [CLS]/[SEP]), restituisce una lista di booleani: True se quel token è il PRIMO
    subword di una parola word-level, False altrimenti.
    I token speciali [CLS] e [SEP] vengono marcati come False (ignorati).
    """
    flags = []
    for tok in subword_tokens:
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            flags.append(False)   # token speciale: non è una parola
        elif tok.startswith("▁") or tok.startswith("Ġ"):
            flags.append(True)    # inizio di una nuova parola (SentencePiece / GPT2-BPE)
        elif not flags:           # primo token assoluto se nessun prefisso (raro)
            flags.append(True)
        else:
            flags.append(False)   # continuazione di subword
    return flags


def subword_preds_to_word_preds(subword_tokens, subword_preds, subword_labels):
    """
    Mappa le predizioni subword → predizioni word-level usando la strategia
    'first-subword': la label del primo subword di ogni parola è la label della parola.

    Restituisce:
        word_preds  -- lista di int con predizioni word-level
        word_trues  -- lista di int con etichette word-level (ground-truth)
    """
    flags = get_word_boundary_flags(subword_tokens)
    word_preds = []
    word_trues = []
    for flag, pred, true in zip(flags, subword_preds, subword_labels):
        if flag and true != -100:   # solo il primo subword di una parola reale
            word_preds.append(pred)
            word_trues.append(true)
    return word_preds, word_trues


def extract_spans_from_word_seq(word_labels, background_label_id):
    """
    Dato un elenco di label word-level, estrae span contigui dello stesso label
    (diverso da background_label_id) come set di tuple (start, end, label_id).
    start/end sono indici 0-based INCLUSIVI sulle parole.
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
    Restituisce (macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, report_lines).
    """
    import numpy as np

    # label ids presenti (esclude background)
    all_ids = sorted(set(list(tp_dict.keys()) + list(fp_dict.keys()) + list(fn_dict.keys())))
    all_ids = [lid for lid in all_ids if lid != background_label_id]

    p_list, r_list, f1_list = [], [], []
    report_lines = []
    header = f"{'Label':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>10}"
    report_lines.append(header)
    report_lines.append("-" * 65)

    for lid in all_ids:
        tp = tp_dict[lid]
        fp = fp_dict[lid]
        fn = fn_dict[lid]
        sup = support_dict[lid]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        name = id2label.get(lid, str(lid))
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
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


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print(f"🔍 DEVICE: {DEVICE}")
    print("\n" + "="*70)
    print("🛠️  Modalità di Test: SOFT TABLE ONLY")
    print("="*70)
    
    # 1. Caricamento Dati
    print(f"\n📥 Caricamento configurazione LABELS...")
    with open(LABEL2DESC_PATH) as f:
        label2desc = json.load(f)
    with open(LABEL2ID_PATH) as f:
        label2id = json.load(f)
    
    id2label = {v: k for k, v in label2id.items()}
    # Ordine garantito per ID
    label_names = [id2label[i] for i in range(len(label2id))]
    
    num_labels = len(label_names)
    print(f"✅ Caricate {num_labels} label (ordinate per ID)")

    print(f"\n📚 Caricamento TEST SET (token-level) da {TEST_PATH}...")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    print(f"✅ Caricati {len(test_records)} esempi di test (token-level)")

    print(f"\n📚 Caricamento TEST SET (span-level) da {TEST_SPAN_PATH}...")
    with open(TEST_SPAN_PATH, "r", encoding="utf-8") as f:
        span_records = json.load(f)
    print(f"✅ Caricati {len(span_records)} esempi di test (span-level)")

    if len(test_records) != len(span_records):
        print(f"⚠️  ATTENZIONE: numero di esempi diverso ({len(test_records)} vs {len(span_records)}). "
              "La valutazione span potrebbe essere inaccurata.")

    # 2. Caricamento Modello
    print("\n" + "="*70)
    print("🔸 CARICAMENTO MODELLO FINE-TUNATO")
    print("="*70)
    
    checkpoint_path = select_checkpoint_interactive(SAVINGS_DIR)
    checkpoint_filename = os.path.basename(checkpoint_path)
    
    print(f"\n📦 Caricamento checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"✅ Checkpoint caricato.")

    print("\n📋 CONTENUTO COMPLETO DEL CHECKPOINT (SAVE DICT):")
    print("=" * 60)
    for key, value in checkpoint.items():
        if key in ['soft_embeddings', 'projection', 'model_state_dict', 'optimizer_state_dict', 'label2id', 'id2label', 'label2desc']:
             if key in ['label2id', 'id2label', 'label2desc']:
                 print(f"🔹 [{key}]: <Dict with {len(value)} items> (Skipped for brevity)")
             else:
                 print(f"🔹 [{key}]: <Tensor/StateDict> (Skipped)")
        elif isinstance(value, dict):
             print(f"🔹 [{key}]:")
             for sub_k, sub_v in value.items():
                 print(f"    - {sub_k}: {sub_v}")
        else:
             print(f"🔹 [{key}]: {value}")
    print("=" * 60)

    # Estrazione iperparametri DAL FILE PT
    hyperparams = checkpoint.get("hyperparameters", {})
    if not hyperparams:
        # Fallback: prova a cercare chiavi comuni al top-level se non c'è la chiave 'hyperparameters'
        common_keys = ["batch_size", "lr", "learning_rate", "epochs", "weight_decay", "temperature", "grad_clip", "warmup_steps", "patience"]
        for k in common_keys:
            if k in checkpoint:
                hyperparams[k] = checkpoint[k]
    
    model = GLiNER.from_pretrained(MODEL_NAME)
    core = model.model
    
    txt_enc = core.token_rep_layer.bert_layer.model
    lbl_enc = core.token_rep_layer.labels_encoder.model
    proj = core.token_rep_layer.labels_projection
    
    # Check presenza soft embeddings
    soft_table = None
    if 'soft_embeddings' in checkpoint:
        print("✅ Trovati 'soft_embeddings' nel checkpoint.")
        # Ricostruiamo la tabella
        weights = checkpoint['soft_embeddings']['weight']
        soft_table = torch.nn.Embedding(weights.size(0), weights.size(1))
        soft_table.load_state_dict(checkpoint['soft_embeddings'])
        soft_table.to(DEVICE)
        
        proj.load_state_dict(checkpoint['projection'])
        print("✅ Soft Table + Projection caricati.")
    else:
        print("❌ ERRORE: 'soft_embeddings' NON trovati nel checkpoint.")
        print("   Questo script supporta solo modalità SOFT TABLE.")
        exit(1)

    txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
    lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)
    
    txt_enc.to(DEVICE)
    lbl_enc.to(DEVICE)
    proj.to(DEVICE)
    
    # 3. Valutazione
    print(f"\n🔍 Valutazione in corso (Soft Table)...")
    
    txt_enc.eval()
    lbl_enc.eval()
    proj.eval()
    soft_table.eval()
    
    y_true_all = []
    y_pred_all = []
    n_skipped = 0

    # --- Span-based accumulatori ---
    from collections import defaultdict
    import numpy as np
    # "background" nel dataset tknlvl corrisponde alla label con id == label2id.get("O", 5)
    background_id = label2id.get("O", 5)
    span_tp = defaultdict(int)
    span_fp = defaultdict(int)
    span_fn = defaultdict(int)
    span_support = defaultdict(int)
    n_span_skipped = 0

    # Label IDs rilevanti (esclude 'O') per progress checkpoint
    all_label_ids_pre = list(range(num_labels))
    o_id_pre = label2id.get("O", -1)
    relevant_label_ids = [i for i in all_label_ids_pre if i != o_id_pre]

    total_records = len(test_records)
    checkpoint_interval = max(1, total_records // 5)
    print(f"\n📊 Mostro progress ogni {checkpoint_interval} record (~20%)")

    with torch.no_grad():
        label_matrix = compute_label_matrix_from_soft_table(soft_table, proj).to(DEVICE)
        
        infer_start_time = time.time()
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
            
            # Collect token-based metrics
            subword_preds_list = preds.tolist()
            for pred, true_label in zip(subword_preds_list, labels):
                if true_label != -100:
                    y_true_all.append(true_label)
                    y_pred_all.append(pred)

            # ---- Span-based evaluation ----
            # Abbiniamo al record span corrispondente (stesso indice)
            if idx < len(span_records):
                span_rec = span_records[idx]
                gt_ner = span_rec.get("ner", [])  # [[start, end, label_id_str], ...]

                # Ground-truth span set (esclude background)
                gt_spans = set()
                for s, e, lbl_str in gt_ner:
                    lbl_int = int(lbl_str)
                    if lbl_int != background_id:
                        gt_spans.add((s, e, lbl_int))
                        span_support[lbl_int] += 1

                # Predizioni word-level dalla sequenza subword
                word_preds, _ = subword_preds_to_word_preds(
                    tokens, subword_preds_list, labels
                )
                pred_spans = extract_spans_from_word_seq(word_preds, background_id)

                # TP / FP / FN
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
                    print(f"\n [{progress:5.1f}%] Macro F1 (No-O): {_mf1:.4f} | Micro F1 (No-O): {_uf1:.4f} | Tokens: {len(y_true_all):,}")

    infer_time = time.time() - infer_start_time
    samples_per_sec = len(test_records) / infer_time if infer_time > 0 else 0
    tokens_per_sec = len(y_true_all) / infer_time if infer_time > 0 else 0

    # 4. Report
    if n_span_skipped > 0:
        print(f"⚠️  {n_span_skipped} esempi saltati nella valutazione span (indice fuori range).")

    all_label_ids = list(range(num_labels))
    
    # Identifica ID della label "O" se presente
    o_label_id = label2id.get("O", -1)
    no_o_label_ids = [lid for lid in all_label_ids if lid != o_label_id]
    
    # --- METRICHE COMPLETE (Con O) ---
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

    # --- METRICHE CLEAN (Senza O) ---
    f1_macro_no_o, f1_micro_no_o = 0.0, 0.0
    if o_label_id != -1 and len(no_o_label_ids) > 0:
        p_ma_no, r_ma_no, f1_macro_no_o, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average="macro", zero_division=0, labels=no_o_label_ids
        )
        p_mi_no, r_mi_no, f1_micro_no_o, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average="micro", zero_division=0, labels=no_o_label_ids
        )
    
    print(f"\n{'='*70}")
    print(f"📊 RISULTATI TEST (SOFT TABLE)")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_filename}")
    print(f"Token valutati: {len(y_true_all):,}")
    print(f"Tempo di inferenza:  {infer_time:.2f} s")
    print(f"Velocità:            {samples_per_sec:.2f} samples/s | {tokens_per_sec:.2f} tokens/s")
    
    print(f"\n🎯 METRICHE AGGREGATE (Tutte le label inclusa 'O'):")
    print(f"  Macro F1:          {f1_macro:.4f}")
    print(f"  Micro F1:          {f1_micro:.4f}")

    if o_label_id != -1:
        print(f"\n🚀 METRICHE CLEAN (Esclusa 'O'):")
        print(f"  Macro F1 (No-O):   {f1_macro_no_o:.4f}")
        print(f"  Micro F1 (No-O):   {f1_micro_no_o:.4f}")

    # ---- Span-based report ----
    span_macro_p, span_macro_r, span_macro_f1, \
    span_micro_p, span_micro_r, span_micro_f1, \
    span_report_lines = compute_span_metrics(
        span_tp, span_fp, span_fn, span_support, id2label, background_id
    )

    print(f"\n{'='*70}")
    print(f"🎯 METRICHE SPAN-BASED (Exact Match, escluso background 'O')")
    print(f"{'='*70}")
    print(f"  Macro P:   {span_macro_p:.4f}  |  Macro R:   {span_macro_r:.4f}  |  Macro F1:   {span_macro_f1:.4f}")
    print(f"  Micro P:   {span_micro_p:.4f}  |  Micro R:   {span_micro_r:.4f}  |  Micro F1:   {span_micro_f1:.4f}")
    print(f"\n📋 REPORT SPAN PER CLASSE:")
    for line in span_report_lines:
        print(line)
    
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
    
    # Salvataggio
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{results_dir}/test_eval_SOFT_TABLE_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Risultati Test - SOFT TABLE\n\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Checkpoint:** {checkpoint_filename}\n")
        
        f.write(f"## Dettagli Training e Configurazione\n")
        
        # 1. Iperparametri Base (lr, batch_size, etc.)
        f.write("### Iperparametri\n")
        if hyperparams:
            for k, v in hyperparams.items():
                f.write(f"- **{k}:** {v}\n")
        else:
            f.write("_Nessun iperparametro di base trovato._\n")

        # 2. Configurazione Aggiuntiva (Parametri Loss, Gamma, Temp, ecc)
        # Recuperiamo direttamente dal checkpoint se presente
        config_data = checkpoint.get("config", {})
        if config_data and isinstance(config_data, dict):
            f.write("\n### Configurazione (Loss/Model)\n")
            for k, v in config_data.items():
                f.write(f"- **{k}:** {v}\n")

        # 3. Info Training (Epoche, Best Val Score)
        train_info = checkpoint.get("training_info", {})
        if train_info and isinstance(train_info, dict):
            f.write("\n### Info Training Run\n")
            for k, v in train_info.items():
                f.write(f"- **{k}:** {v}\n")
        
        f.write(f"\n## Metriche aggregate (Standard)\n")
        f.write(f"- **Token valutati:** {len(y_true_all):,}\n")
        f.write(f"- **Tempo Inferenza:** {infer_time:.2f} s\n")
        f.write(f"- **Velocità:** {samples_per_sec:.2f} samples/s | {tokens_per_sec:.2f} tokens/s\n\n")
        f.write(f"- **Macro F1:** {f1_macro:.4f}\n")
        f.write(f"- **Micro F1:** {f1_micro:.4f}\n")

        if o_label_id != -1:
             f.write(f"\n## Metriche Clean (No 'O')\n")
             f.write(f"- **Macro F1 (No-O):** {f1_macro_no_o:.4f}\n")
             f.write(f"- **Micro F1 (No-O):** {f1_micro_no_o:.4f}\n")

        f.write(f"\n## Report Token-Level per classe\n```\n{class_report}\n```\n")

        f.write(f"\n## Metriche Span-Based (Exact Match, escluso background 'O')\n")
        f.write(f"| Metric | Precision | Recall | F1 |\n")
        f.write(f"|------|-----------|--------|-----|\n")
        f.write(f"| **Macro** | {span_macro_p:.4f} | {span_macro_r:.4f} | **{span_macro_f1:.4f}** |\n")
        f.write(f"| **Micro** | {span_micro_p:.4f} | {span_micro_r:.4f} | **{span_micro_f1:.4f}** |\n")
        f.write(f"\n### Report Span per Classe\n```\n")
        for line in span_report_lines:
            f.write(line + "\n")
        f.write("```\n")
    
    print(f"\n💾 Risultati salvati in: {filename}")
    print(f"✅ Test completato!")
