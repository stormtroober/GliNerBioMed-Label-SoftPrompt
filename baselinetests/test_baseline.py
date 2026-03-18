"""
test_baseline.py — Valutazione span-based del modello GLiNER baseline (no fine-tuning).

Testa il modello baseline (bi-encoder o mono-encoder) su test_dataset_span_bi.json
(bi) o test_dataset_span_mono.json (mono) con la stessa utility di evaluate di
finetune/test_finetuned.py:
  - span exact-match (char → token mapping)
  - RUN 1 — NAMES:        label brevi ("protein", "dna", ...) come entity type
  - RUN 2 — DESCRIPTIONS: descrizioni estese da label2desc.json come entity type
                           (per il mono-encoder le descrizioni sono troncate
                            automaticamente al budget disponibile di token)

Uso:
    python test_baseline.py
    python test_baseline.py --encoder_type bi --dataset jnlpba --batch_size 8
"""

import os
import datetime
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from gliner import GLiNER

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIRS = {
    "jnlpba": "../dataset",
    "bc5cdr": "../dataset_bc5cdr",
}
LABEL2DESC_FILENAME = "label2desc.json"
TEST_RESULTS_DIR = "results"

# ==========================================
# SPAN-BASED METRICS UTILITY
# (stessa logica di finetune/test_finetuned.py)
# ==========================================

def calculate_metrics(dataset, model, label_list, batch_size=8):
    """
    Calcola metriche span-based (exact match char→token) per un dataset span.

    Args:
        dataset:    lista di record {"tokenized_text": [...], "ner": [[s,e,lbl], ...]}
                    con lbl già in formato stringa (es. "chemical")
        model:      istanza GLiNER già caricata
        label_list: lista di stringhe label da valutare (es. ["chemical","disease"])
        batch_size: dimensione batch per l'inferenza

    Returns:
        (macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, report_str,
         total_inference_time, avg_iter_per_sec, samples_per_sec)
    """
    import gc
    import time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    label_list = sorted(list(set(label_list)))
    n_samples = len(dataset)
    print(f"\nEvaluating on {n_samples} samples with {len(label_list)} labels: {label_list}")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    batch_indices = list(range(0, n_samples, batch_size))
    n_batches = len(batch_indices)

    inference_start = time.perf_counter()

    for i in tqdm(batch_indices, desc="Evaluating"):
        batch_items = dataset[i:i + batch_size]
        # GLiNER si aspetta testo stringa, non token separati
        batch_texts = [" ".join(d["tokenized_text"]) for d in batch_items]

        with torch.no_grad():
            if hasattr(model, "inference"):
                batch_preds = model.inference(batch_texts, label_list, threshold=0.5)
            elif hasattr(model, "batch_predict_entities"):
                batch_preds = model.batch_predict_entities(batch_texts, label_list, threshold=0.5)
            else:
                batch_preds = [model.predict_entities(t, label_list, threshold=0.5) for t in batch_texts]

        for idx, item in enumerate(batch_items):
            # Ground-truth spans: solo le label in label_list
            gt_spans = set()
            for s, e, lbl in item["ner"]:
                if lbl in label_list:
                    gt_spans.add((lbl, s, e))
                    support[lbl] += 1

            # Mappa carattere → indice token per ricostruire span token-level
            tokens = item["tokenized_text"]
            char_to_token = {}
            cursor = 0
            for t_i, token in enumerate(tokens):
                for c in range(cursor, cursor + len(token)):
                    char_to_token[c] = t_i
                cursor += len(token) + 1  # +1 per lo spazio tra token

            # Predizioni span
            preds_raw = batch_preds[idx]
            pred_spans = set()
            for p in preds_raw:
                label = p["label"]
                if label not in label_list:
                    continue
                p_start = p["start"]
                p_end = p["end"]
                if p_start in char_to_token and (p_end - 1) in char_to_token:
                    t_start = char_to_token[p_start]
                    t_end = char_to_token[p_end - 1]
                    pred_spans.add((label, t_start, t_end))

            tps = pred_spans & gt_spans
            fps = pred_spans - gt_spans
            fns = gt_spans - pred_spans

            for lbl, s, e in tps: tp[lbl] += 1
            for lbl, s, e in fps: fp[lbl] += 1
            for lbl, s, e in fns: fn[lbl] += 1

    total_inference_time = time.perf_counter() - inference_start
    avg_iter_per_sec = n_batches / total_inference_time if total_inference_time > 0 else 0.0
    samples_per_sec  = n_samples / total_inference_time if total_inference_time > 0 else 0.0

    print(f"\n⏱  Inference time : {total_inference_time:.2f}s")
    print(f"   Avg iter/sec   : {avg_iter_per_sec:.2f}")
    print(f"   Samples/sec    : {samples_per_sec:.2f}")

    # --- Per-label report ---
    valid_labels = sorted(set(list(tp.keys()) + list(fn.keys())))
    p_list, r_list, f1_list = [], [], []
    report_lines = []
    header = f"{'Label':<30} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Supp.':<8}"
    sep = "-" * 75
    report_lines += [header, sep]
    print(f"\n### Per-Label Performance")
    print(header)
    print(sep)

    for lbl in valid_labels:
        t  = tp[lbl]
        f_p = fp[lbl]
        f_n = fn[lbl]
        p  = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r  = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        line = f"{lbl:<30} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {support[lbl]:<8}"
        print(line)
        report_lines.append(line)

    macro_p  = float(np.mean(p_list))  if p_list  else 0.0
    macro_r  = float(np.mean(r_list))  if r_list  else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"\n### Performance Summary")
    print(f"| Average | Precision | Recall | F1-Score |")
    print(f"|:--------|----------:|-------:|---------:|")
    print(f"| **Macro** | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro** | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

    return (
        macro_p, macro_r, macro_f1,
        micro_p, micro_r, micro_f1,
        "\n".join(report_lines),
        total_inference_time, avg_iter_per_sec, samples_per_sec,
    )


def convert_ids_to_labels(dataset, id2label, filter_o=True):
    """
    Converte gli ID numerici delle label in nomi stringa.
    Se filter_o=True, esclude i record con label 'O'.
    Restituisce un NUOVO dataset (non modifica l'originale).
    """
    new_dataset = []
    for item in dataset:
        new_ner = []
        for start, end, label_id in item["ner"]:
            label_name = id2label.get(str(label_id))
            if label_name is None:
                continue
            if filter_o and label_name == "O":
                continue
            new_ner.append([start, end, label_name])
        new_item = dict(item)
        new_item["ner"] = new_ner
        new_dataset.append(new_item)
    return new_dataset


def compute_desc_max_tokens(model, label2desc, dataset, encoder_type,
                             model_max_length=512, text_budget_tokens=150,
                             overhead_tokens=16):
    """
    Calcola quanti token sono disponibili per ciascuna descrizione nel mono-encoder.

    Per il mono-encoder GLiNER il template della sequenza è:
        [CLS] <label1> [SEP] <label2> [SEP] ... [SEP] <token1> ... <tokenN> [SEP]
    Tutto deve stare in model_max_length token.

    Per il bi-encoder le label vengono encodate separatamente dal testo,
    quindi non c'è limite pratico → ritorna None (nessun troncamento).

    Args:
        model:              istanza GLiNER già caricata
        label2desc:         dict {nome_breve: descrizione}
        dataset:            dataset di test (per stimare la lunghezza media del testo)
        encoder_type:       "bi" | "mono"
        model_max_length:   max token del modello (default 512)
        text_budget_tokens: token riservati al testo (conservativo)
        overhead_tokens:    token overhead (CLS, SEP per ogni label, ecc.)

    Returns:
        max_tokens_per_desc (int) oppure None se nessun troncamento necessario
    """
    if encoder_type == "bi":
        print("  ℹ️  Bi-encoder: le label sono processate separatamente → nessun troncamento.")
        return None

    num_labels = len([k for k in label2desc if k != "O"])
    if num_labels == 0:
        return None

    # Token stimati per i separatori (ogni label aggiunge ~2 token: [SEP] + whitespace)
    sep_budget = num_labels * 2

    # Budget disponibile per TUTTE le descrizioni
    available = model_max_length - text_budget_tokens - overhead_tokens - sep_budget
    available = max(available, num_labels * 5)  # almeno 5 token per label

    # Budget per ciascuna descrizione
    per_desc = available // num_labels

    print(f"  ℹ️  Mono-encoder max_length={model_max_length}: "
          f"{num_labels} label × {per_desc} token/label "
          f"(text_budget={text_budget_tokens}, overhead={overhead_tokens+sep_budget})")
    return per_desc


def truncate_desc_by_tokens(desc, tokenizer, max_tokens):
    """
    Trunca una descrizione a max_tokens token usando il tokenizer del modello.
    Aggiunge '...' se la stringa è stata troncata.
    """
    tokens = tokenizer.encode(desc, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return desc
    truncated_ids = tokens[:max_tokens]
    truncated_str = tokenizer.decode(truncated_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    return truncated_str.rstrip() + "..."


def calculate_metrics_with_desc(dataset, model, label2desc, batch_size=8,
                                max_tokens_per_desc=None):
    """
    Come calculate_metrics, ma usa le DESCRIZIONI estese come entity type passato a GLiNER.
    Il modello restituirà predizioni con label = descrizione;
    queste vengono rimappate al nome breve per il confronto con i GT span.

    Args:
        dataset:             lista di record {"tokenized_text":[...], "ner":[[s,e,nome_breve],...]} 
                             (già convertito con convert_ids_to_labels, filter_o=True)
        model:               istanza GLiNER
        label2desc:          dict {nome_breve: descrizione_testuale}
        batch_size:          int
        max_tokens_per_desc: int oppure None. Se specificato, troncamento automatico
                             delle descrizioni via tokenizer del modello.

    Returns:
        (macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, report_str,
         total_inference_time, avg_iter_per_sec, samples_per_sec)
    """
    import gc
    import time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # --- Troncamento descrizioni se richiesto ---
    name_to_desc_orig = {k: v for k, v in label2desc.items() if k != "O"}
    if max_tokens_per_desc is not None:
        tokenizer = model.data_processor.transformer_tokenizer
        name_to_desc = {}
        print(f"\n  📐 Troncamento descrizioni a max {max_tokens_per_desc} token/label:")
        for name, desc in name_to_desc_orig.items():
            truncated = truncate_desc_by_tokens(desc, tokenizer, max_tokens_per_desc)
            n_orig = len(tokenizer.encode(desc, add_special_tokens=False))
            n_trunc = len(tokenizer.encode(truncated, add_special_tokens=False))
            was_cut = "✂️" if n_trunc < n_orig else "✅"
            print(f"    {was_cut} [{name}]: {n_orig} → {n_trunc} token")
            name_to_desc[name] = truncated
    else:
        name_to_desc = name_to_desc_orig

    # Mapping bidirezionale desc ↔ nome breve (usato sia per l'input che per il match)
    desc_to_name = {v: k for k, v in name_to_desc.items()}
    desc_list  = sorted(name_to_desc.values())   # lista descrizioni (eventualmente troncate)
    label_list = sorted(name_to_desc.keys())     # nomi brevi, per i GT

    n_samples = len(dataset)
    print(f"\nEvaluating on {n_samples} samples — DESCRIPTION mode")
    print(f"Using {len(desc_list)} descriptions (one per label, excluding 'O')")

    tp       = defaultdict(int)
    fp       = defaultdict(int)
    fn       = defaultdict(int)
    support  = defaultdict(int)

    batch_indices = list(range(0, n_samples, batch_size))
    n_batches = len(batch_indices)

    inference_start = time.perf_counter()

    for i in tqdm(batch_indices, desc="Evaluating (desc)"):
        batch_items = dataset[i:i + batch_size]
        batch_texts = [" ".join(d["tokenized_text"]) for d in batch_items]

        with torch.no_grad():
            if hasattr(model, "inference"):
                batch_preds = model.inference(batch_texts, desc_list, threshold=0.5)
            elif hasattr(model, "batch_predict_entities"):
                batch_preds = model.batch_predict_entities(batch_texts, desc_list, threshold=0.5)
            else:
                batch_preds = [model.predict_entities(t, desc_list, threshold=0.5) for t in batch_texts]

        for idx, item in enumerate(batch_items):
            # GT usa nomi brevi
            gt_spans = set()
            for s, e, lbl in item["ner"]:
                if lbl in label_list:
                    gt_spans.add((lbl, s, e))
                    support[lbl] += 1

            # char → token
            tokens = item["tokenized_text"]
            char_to_token = {}
            cursor = 0
            for t_i, token in enumerate(tokens):
                for c in range(cursor, cursor + len(token)):
                    char_to_token[c] = t_i
                cursor += len(token) + 1

            # Predizioni: label è la descrizione (eventualmente troncata) → rimappa al nome breve
            preds_raw = batch_preds[idx]
            pred_spans = set()
            for p in preds_raw:
                desc = p["label"]
                short_name = desc_to_name.get(desc)
                if short_name is None:
                    continue          # descrizione non riconosciuta
                p_start = p["start"]
                p_end   = p["end"]
                if p_start in char_to_token and (p_end - 1) in char_to_token:
                    t_start = char_to_token[p_start]
                    t_end   = char_to_token[p_end - 1]
                    pred_spans.add((short_name, t_start, t_end))

            tps = pred_spans & gt_spans
            fps = pred_spans - gt_spans
            fns = gt_spans - pred_spans

            for lbl, s, e in tps: tp[lbl] += 1
            for lbl, s, e in fps: fp[lbl] += 1
            for lbl, s, e in fns: fn[lbl] += 1

    total_inference_time = time.perf_counter() - inference_start
    avg_iter_per_sec = n_batches / total_inference_time if total_inference_time > 0 else 0.0
    samples_per_sec  = n_samples / total_inference_time if total_inference_time > 0 else 0.0

    print(f"\n⏱  Inference time : {total_inference_time:.2f}s")
    print(f"   Avg iter/sec   : {avg_iter_per_sec:.2f}")
    print(f"   Samples/sec    : {samples_per_sec:.2f}")

    # Report
    valid_labels = sorted(set(list(tp.keys()) + list(fn.keys())))
    p_list, r_list, f1_list = [], [], []
    report_lines = []
    header = f"{'Label':<30} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Supp.':<8}"
    sep    = "-" * 75
    report_lines += [header, sep]
    print(f"\n### Per-Label Performance (description mode)")
    print(header)
    print(sep)

    for lbl in valid_labels:
        t   = tp[lbl]
        f_p = fp[lbl]
        f_n = fn[lbl]
        p   = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r   = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        line = f"{lbl:<30} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {support[lbl]:<8}"
        print(line)
        report_lines.append(line)

    macro_p  = float(np.mean(p_list))  if p_list  else 0.0
    macro_r  = float(np.mean(r_list))  if r_list  else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"\n### Performance Summary (description mode)")
    print(f"| Average | Precision | Recall | F1-Score |")
    print(f"|:--------|----------:|-------:|---------:|")
    print(f"| **Macro** | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro** | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

    return (
        macro_p, macro_r, macro_f1,
        micro_p, micro_r, micro_f1,
        "\n".join(report_lines),
        total_inference_time, avg_iter_per_sec, samples_per_sec,
    )


# ==========================================
# INTERACTIVE HELPERS
# ==========================================

def interactive_select_encoder():
    print("\n" + "=" * 60)
    print("🔧 BASELINE ENCODER TYPE")
    print("=" * 60)
    print("  [1] bi    — Bi-Encoder  (Ihor/gliner-biomed-bi-small-v1.0)")
    print("  [2] mono  — Mono-Encoder (urchade/gliner_small-v2.1)")
    print("=" * 60)
    while True:
        choice = input("Select encoder type [1-2]: ").strip()
        if choice == "1":
            return "bi"
        elif choice == "2":
            return "mono"
        print("Invalid choice. Enter 1 or 2.")


def interactive_select_dataset():
    print("\n" + "=" * 60)
    print("📊 DATASET")
    print("=" * 60)
    print("  [1] jnlpba  — JNLPBA  (../dataset)")
    print("  [2] bc5cdr  — BC5CDR  (../dataset_bc5cdr)")
    print("=" * 60)
    while True:
        choice = input("Select dataset [1-2]: ").strip()
        if choice == "1":
            return "jnlpba"
        elif choice == "2":
            return "bc5cdr"
        print("Invalid choice. Enter 1 or 2.")


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Span-based evaluation of the GLiNER baseline (no fine-tuning)."
    )
    parser.add_argument("--encoder_type", type=str, default=None, choices=["bi", "mono"])
    parser.add_argument("--dataset",      type=str, default=None, choices=["jnlpba", "bc5cdr"])
    parser.add_argument("--batch_size",   type=int, default=8)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Selezioni interattive ---
    encoder_type = args.encoder_type or interactive_select_encoder()
    dataset_name = args.dataset      or interactive_select_dataset()

    # --- Modello ---
    if encoder_type == "bi":
        MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
    else:
        MODEL_NAME = "urchade/gliner_small-v2.1"

    # --- Path dataset ---
    dataset_dir   = os.path.join(script_dir, DATASET_DIRS[dataset_name])
    # Il bi-encoder usa test_dataset_span_bi.json (span già in formato char-level compatibile)
    # Il mono-encoder usa test_dataset_span_mono.json
    if encoder_type == "bi":
        test_path = os.path.join(dataset_dir, "test_dataset_span_bi.json")
    else:
        test_path = os.path.join(dataset_dir, "test_dataset_span_mono.json")
    label2id_path = os.path.join(dataset_dir, "label2id.json")
    label2desc_path = os.path.join(dataset_dir, LABEL2DESC_FILENAME)

    print("\n" + "=" * 60)
    print("📋 BASELINE TEST CONFIGURATION")
    print("=" * 60)
    print(f"  Model:        {MODEL_NAME}")
    print(f"  Encoder:      {encoder_type}")
    print(f"  Dataset:      {dataset_name}")
    print(f"  Test file:    {test_path}")
    print(f"  Label2id:     {label2id_path}")
    print(f"  Label2desc:   {label2desc_path}")
    print(f"  Batch size:   {args.batch_size}")
    print("=" * 60)

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Aborted.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    # --- Caricamento modello ---
    print(f"\nLoading model: {MODEL_NAME}")
    try:
        model = GLiNER.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Caricamento dati ---
    print(f"\nLoading test data from {test_path}...")
    with open(test_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)
    print(f"✅ {len(raw_dataset)} test examples loaded.")

    print(f"Loading label map from {label2id_path}...")
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    print(f"✅ {len(label2id)} labels: {sorted(label2id.keys())}")

    print(f"Loading descriptions from {label2desc_path}...")
    with open(label2desc_path, "r", encoding="utf-8") as f:
        label2desc = json.load(f)
    print(f"✅ {len(label2desc)} descriptions loaded.")

    # ================================================================
    # EVALUATION 1: NAMES — label brevi ("protein", "dna", ...)
    # ================================================================
    print("\n" + "=" * 60)
    print("🟢 EVALUATION 1 — LABEL NAMES (Excluding 'O')")
    print("=" * 60)

    dataset_clean = convert_ids_to_labels(raw_dataset, id2label, filter_o=True)
    labels_clean  = sorted({lbl for item in dataset_clean for _, _, lbl in item["ner"]})

    (
        mac_p_n, mac_r_n, mac_f1_n,
        mic_p_n, mic_r_n, mic_f1_n,
        report_names,
        inf_time_n, iter_sec_n, samp_sec_n,
    ) = calculate_metrics(
        dataset_clean, model, label_list=labels_clean, batch_size=args.batch_size
    )

    # ================================================================
    # EVALUATION 2: DESCRIPTIONS — testi estesi da label2desc.json
    # ================================================================
    print("\n" + "=" * 60)
    print("🔵 EVALUATION 2 — LABEL DESCRIPTIONS (Excluding 'O')")
    print("=" * 60)

    # Calcola il budget di token per ciascuna descrizione
    # (necessario solo per mono-encoder, dove label+testo devono stare in 512 token)
    print("\nCalcolo budget token per le descrizioni...")
    max_tokens_per_desc = compute_desc_max_tokens(
        model=model,
        label2desc=label2desc,
        dataset=dataset_clean,
        encoder_type=encoder_type,
    )

    # dataset_clean già preparato sopra (stessi GT span, stessi nomi brevi)
    (
        mac_p_d, mac_r_d, mac_f1_d,
        mic_p_d, mic_r_d, mic_f1_d,
        report_desc,
        inf_time_d, iter_sec_d, samp_sec_d,
    ) = calculate_metrics_with_desc(
        dataset_clean, model, label2desc=label2desc,
        batch_size=args.batch_size,
        max_tokens_per_desc=max_tokens_per_desc,
    )

    # ================================================================
    # SALVATAGGIO
    # ================================================================
    results_dir = os.path.join(script_dir, TEST_RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        results_dir,
        f"eval_BASELINE_{dataset_name}_{encoder_type}enc_{timestamp}.md"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Baseline Evaluation — Span-Based\n\n")

        f.write("## 🔍 Configuration\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| **Model** | `{MODEL_NAME}` |\n")
        f.write(f"| **Encoder** | `{encoder_type}` |\n")
        f.write(f"| **Dataset** | `{dataset_name}` |\n")
        f.write(f"| **Test file** | `{os.path.basename(test_path)}` |\n")
        f.write(f"| **Test samples** | `{len(dataset_clean)}` |\n")
        f.write(f"| **Batch size** | `{args.batch_size}` |\n")
        f.write(f"| **Timestamp** | `{timestamp}` |\n\n")

        f.write("## 🟢 Evaluation 1 — Label Names (Excluding 'O')\n")
        f.write(f"Entity types used: {labels_clean}\n\n")
        f.write("| Metric | Precision | Recall | F1 |\n|---|---|---|---|\n")
        f.write(f"| **Macro** | {mac_p_n:.4f} | {mac_r_n:.4f} | **{mac_f1_n:.4f}** |\n")
        f.write(f"| **Micro** | {mic_p_n:.4f} | {mic_r_n:.4f} | **{mic_f1_n:.4f}** |\n\n")
        f.write("| Timing | Value |\n|---|---|\n")
        f.write(f"| **Total inference time** | `{inf_time_n:.2f} s` |\n")
        f.write(f"| **Avg iterations/sec** | `{iter_sec_n:.2f}` |\n")
        f.write(f"| **Samples/sec** | `{samp_sec_n:.2f}` |\n\n")
        f.write(f"```\n{report_names}\n```\n\n")

        desc_trunc_note = (
            f"`{max_tokens_per_desc}` token/label (troncato per mono-encoder)"
            if max_tokens_per_desc is not None
            else "nessun troncamento (bi-encoder, label processate separatamente)"
        )
        f.write("## 🔵 Evaluation 2 — Label Descriptions (Excluding 'O')\n")
        f.write(f"Entity types used: extended descriptions from label2desc.json\n")
        f.write(f"Description truncation: {desc_trunc_note}\n\n")
        f.write("| Metric | Precision | Recall | F1 |\n|---|---|---|---|\n")
        f.write(f"| **Macro** | {mac_p_d:.4f} | {mac_r_d:.4f} | **{mac_f1_d:.4f}** |\n")
        f.write(f"| **Micro** | {mic_p_d:.4f} | {mic_r_d:.4f} | **{mic_f1_d:.4f}** |\n\n")
        f.write("| Timing | Value |\n|---|---|\n")
        f.write(f"| **Total inference time** | `{inf_time_d:.2f} s` |\n")
        f.write(f"| **Avg iterations/sec** | `{iter_sec_d:.2f}` |\n")
        f.write(f"| **Samples/sec** | `{samp_sec_d:.2f}` |\n\n")
        f.write(f"```\n{report_desc}\n```\n")

    print(f"\n💾 Results saved to: {filename}")
    print("✅ Baseline test completed!")


if __name__ == "__main__":
    main()
