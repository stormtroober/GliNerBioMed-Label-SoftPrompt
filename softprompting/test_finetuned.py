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

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#PATH_DATASET = "../dataset"
PATH_DATASET = "../dataset_bc5cdr"

TEST_PATH = PATH_DATASET + "/test_dataset_tknlvl_bi.json"
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

    print(f"\n📚 Caricamento TEST SET da {TEST_PATH}...")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    print(f"✅ Caricati {len(test_records)} esempi di test")

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
    
    with torch.no_grad():
        label_matrix = compute_label_matrix_from_soft_table(soft_table, proj).to(DEVICE)
        
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
    
    # 4. Report
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
    
    print(f"\n🎯 METRICHE AGGREGATE (Tutte le label inclusa 'O'):")
    print(f"  Macro F1:          {f1_macro:.4f}")
    print(f"  Micro F1:          {f1_micro:.4f}")

    if o_label_id != -1:
        print(f"\n🚀 METRICHE CLEAN (Esclusa 'O'):")
        print(f"  Macro F1 (No-O):   {f1_macro_no_o:.4f}")
        print(f"  Micro F1 (No-O):   {f1_micro_no_o:.4f}")
    
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
        f.write(f"- **Macro F1:** {f1_macro:.4f}\n")
        f.write(f"- **Micro F1:** {f1_micro:.4f}\n")

        if o_label_id != -1:
             f.write(f"\n## Metriche Clean (No 'O')\n")
             f.write(f"- **Macro F1 (No-O):** {f1_macro_no_o:.4f}\n")
             f.write(f"- **Micro F1 (No-O):** {f1_micro_no_o:.4f}\n")

        f.write(f"\n## Report per classe\n```\n{class_report}\n```\n")
    
    print(f"\n💾 Risultati salvati in: {filename}")
    print(f"✅ Test completato!")
