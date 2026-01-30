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
from collections import Counter
import os
from datetime import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "../dataset/test_dataset_tknlvl_bi.json"
LABEL2DESC_PATH = "../dataset/label2desc.json"
LABEL2ID_PATH = "../dataset/label2id.json"
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

print(f"📚 Caricamento test set da {TEST_PATH}...")
with open(TEST_PATH, "r") as f:
    test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

# ==========================================================
# 2️⃣ FUNZIONI HELPER
# ==========================================================
def select_checkpoint_interactive(savings_dir="savings"):
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
    print("📦 SELEZIONE CHECKPOINT")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        import datetime
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

def compute_label_matrix(label2desc, lbl_tok, lbl_enc, proj):
    """Embedda le descrizioni con encoder + projection."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = lbl_enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)

def evaluate_model(txt_enc, lbl_enc, proj, txt_tok, lbl_tok, model_name, checkpoint_info=None):
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
            # Ricrea sempre input_ids e attention_mask dai tokens
            labels = record["labels"]
            tokens = record["tokens"]  # includono [CLS]/[SEP]

            ids = txt_tok.convert_tokens_to_ids(tokens)

            # Troncamento sicuro su max length del modello, preservando primo e ultimo token
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
    
    # Salva risultati con datetime nel nome
    os.makedirs("testresults", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{model_name.replace(' ', '_').replace('-', '_')}_{timestamp}.md"
    
    with open(f'testresults/{filename}', "w") as f:
        f.write(f"# Risultati - {model_name}\n\n")
        f.write(f"**Data test:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Se abbiamo info dal checkpoint, includiamo tutti i dettagli
        if checkpoint_info:
            f.write(f"## Informazioni Checkpoint\n\n")
            f.write(f"**Checkpoint path:** `{checkpoint_info.get('checkpoint_path', 'N/A')}`\n\n")
            
            # Dataset info
            f.write(f"### Dataset di Training\n\n")
            f.write(f"- **Nome dataset:** {checkpoint_info.get('dataset_name', 'N/A')}\n")
            f.write(f"- **Path dataset:** `{checkpoint_info.get('dataset_path', 'N/A')}`\n")
            f.write(f"- **Dimensione dataset:** {checkpoint_info.get('dataset_size', 'N/A')} samples\n\n")
            
            # Iperparametri
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
            
            # Training info
            training_info = checkpoint_info.get('training_info', {})
            if training_info:
                f.write(f"### Risultati Training\n\n")
                f.write(f"- **Best Loss:** {training_info.get('best_loss', 'N/A'):.4f}\n")
                f.write(f"- **Best Epoch:** {training_info.get('best_epoch', 'N/A')}\n")
                training_time = training_info.get('total_training_time_seconds', 0)
                f.write(f"- **Tempo Training:** {training_time:.1f}s ({training_time/60:.1f} min)\n")
                f.write(f"- **Modello Base:** `{training_info.get('model_name', 'N/A')}`\n\n")
        
        f.write(f"## Risultati Test\n\n")
        f.write(f"**Token valutati:** {len(y_true_all)}\n\n")
        f.write(f"### Metriche aggregate\n\n")
        f.write(f"| Metrica | Valore |\n")
        f.write(f"|---------|--------|\n")
        f.write(f"| Macro F1 | {f1_macro:.4f} |\n")
        f.write(f"| Micro F1 | {f1_micro:.4f} |\n")
        f.write(f"| Precision (macro) | {prec_macro:.4f} |\n")
        f.write(f"| Recall (macro) | {rec_macro:.4f} |\n")
        f.write(f"| Precision (micro) | {prec_micro:.4f} |\n")
        f.write(f"| Recall (micro) | {rec_micro:.4f} |\n\n")
        
        # Distribuzione predizioni vs ground truth
        f.write(f"### Distribuzione Predizioni\n\n")
        f.write(f"| Label | Predizioni | Ground Truth |\n")
        f.write(f"|-------|------------|-------------|\n")
        for label_id in sorted(true_counts.keys(), key=lambda x: true_counts[x], reverse=True):
            label_name = id2label[label_id]
            f.write(f"| {label_name} | {pred_counts.get(label_id, 0)} | {true_counts[label_id]} |\n")
        f.write(f"\n")
        
        f.write(f"### Report per classe\n\n```\n{class_report}\n```\n")
    
    print(f"💾 Risultati salvati in: testresults/{filename}")
    
    return {
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'micro_precision': prec_micro,
        'micro_recall': rec_micro
    }

# ==========================================================
# 3️⃣ VALUTAZIONE MODELLO FINE-TUNATO
# ==========================================================
print("\n" + "="*60)
print("🔸 MODELLO FINE-TUNATO")
print("="*60)

# Selezione interattiva checkpoint
CHECKPOINT_PATH = select_checkpoint_interactive("savings")

# Carica modello base
model_ft = GLiNER.from_pretrained(MODEL_NAME)
core_ft = model_ft.model

txt_enc_ft = core_ft.token_rep_layer.bert_layer.model
lbl_enc_ft = core_ft.token_rep_layer.labels_encoder.model
proj_ft = core_ft.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc_ft.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc_ft.config._name_or_path)

# Carica checkpoint
print(f"📦 Caricamento checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Estrai informazioni dal checkpoint (nuovo formato)
checkpoint_info = {
    'checkpoint_path': CHECKPOINT_PATH,
    'dataset_name': checkpoint.get('dataset_name', 'N/A'),
    'dataset_path': checkpoint.get('dataset_path', 'N/A'),
    'dataset_size': checkpoint.get('dataset_size', 'N/A'),
    'hyperparameters': checkpoint.get('hyperparameters', {}),
    'training_info': checkpoint.get('training_info', {}),
}

# Stampa info checkpoint
print(f"📊 Dataset di training: {checkpoint_info['dataset_name']} ({checkpoint_info['dataset_size']} samples)")
if checkpoint_info['hyperparameters']:
    hp = checkpoint_info['hyperparameters']
    print(f"⚙️  Hyperparameters: bs={hp.get('batch_size')}, lr={hp.get('learning_rate')}, epochs={hp.get('epochs')}")

lbl_enc_ft.load_state_dict(checkpoint['label_encoder_state_dict'])
proj_ft.load_state_dict(checkpoint['projection_state_dict'])

txt_enc_ft.to(DEVICE)
lbl_enc_ft.to(DEVICE)
proj_ft.to(DEVICE)

results_ft = evaluate_model(txt_enc_ft, lbl_enc_ft, proj_ft, txt_tok, lbl_tok, "Fine_tuned", checkpoint_info)

print("\n✅ Valutazione completata!")