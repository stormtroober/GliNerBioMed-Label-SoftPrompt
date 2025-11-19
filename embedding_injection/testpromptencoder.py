# -*- coding: utf-8 -*-
"""
Valutazione modello Bi-GLiNER con Prompt Encoder esterno.
Ricostruisce la pipeline:
Descrizioni -> Prompt Encoder (Trained) -> Label Encoder (Frozen) -> Projection (Frozen)
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from torch import nn
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter
import os
import datetime
from tqdm import tqdm

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "../dataset/test_dataset_tokenlevel.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
SAVINGS_DIR = "savings"

# ==========================================================
# 1️⃣ DEFINIZIONE PROMPT ENCODER (Deve combaciare col training)
# ==========================================================
class PromptEncoder(nn.Module):
    """
    Lo stesso modulo usato nel training per caricare i pesi salvati.
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Qui non serve inizializzare con i pesi originali perché tanto
        # caricheremo lo state_dict salvato subito dopo.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, input_ids):
        return self.embedding(input_ids)

# ==========================================================
# UTILITÀ
# ==========================================================
def select_checkpoint_interactive(savings_dir=SAVINGS_DIR):
    """Mostra menu interattivo per selezionare checkpoint."""
    if not os.path.exists(savings_dir):
        print(f"❌ Directory {savings_dir} non trovata.")
        exit()
        
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
    print("📦 SELEZIONE CHECKPOINT (PROMPT ENCODER)")
    print("="*60)
    
    for i, info in enumerate(checkpoints_info, 1):
        date_str = datetime.datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {info['name']}")
        print(f"   📅 {date_str} | 💾 {info['size_mb']:.1f} MB")
    
    while True:
        try:
            choice = input(f"\n👉 Seleziona checkpoint (1-{len(checkpoints_info)}) [default: 1]: ").strip()
            if choice == "": choice = 1
            else: choice = int(choice)
            
            if 1 <= choice <= len(checkpoints_info):
                selected = checkpoints_info[choice - 1]
                print(f"✅ Selezionato: {selected['name']}")
                return selected['path']
            else: print(f"❌ Numero non valido.")
        except: print("❌ Input non valido.")

def truncate_tokens_safe(tokens, tokenizer, max_len=None):
    if max_len is None: max_len = tokenizer.model_max_length
    if len(tokens) <= max_len: return tokens
    if tokens[0] == tokenizer.cls_token and tokens[-1] == tokenizer.sep_token and max_len >= 2:
        return [tokens[0]] + tokens[1:max_len-1] + [tokens[-1]]
    return tokens[:max_len]

# ==========================================================
# 🚀 COSTRUZIONE MATRICE LABEL (CORE LOGIC)
# ==========================================================
def build_optimized_label_matrix(prompt_enc, lbl_enc, proj, lbl_tok, label_names, label2desc):
    """
    Genera la matrice delle label usando la pipeline addestrata:
    Prompt Encoder -> Label Encoder (Inputs Embeds) -> Pooling -> Projection
    """
    print("⚙️  Generazione matrice label ottimizzata...")
    
    desc_texts = [label2desc[name] for name in label_names]
    
    # 1. Tokenizzazione Descrizioni
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    with torch.no_grad():
        # 2. Prompt Encoder (Soft Embeddings)
        soft_embeds = prompt_enc(input_ids) # [num_labels, seq, 384]
        
        # 3. Injection nel Label Encoder
        outputs = lbl_enc(inputs_embeds=soft_embeds, attention_mask=attention_mask)
        
        # 4. Pooling (Weighted Mean)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 5. Projection
        projected = proj(pooled)
        
        # 6. Normalizzazione
        label_matrix = F.normalize(projected, dim=-1)
        
    return label_matrix

# ==========================================================
# MAIN FLOW
# ==========================================================

# 1. Caricamento Dati
print("📥 Caricamento mappe...")
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label_names)

print(f"\n📚 Caricamento TEST SET da {TEST_PATH}...")
with open(TEST_PATH, "r", encoding="utf-8") as f: test_records = json.load(f)
print(f"✅ Caricati {len(test_records)} esempi di test")

# 2. Caricamento Modello Base (Blackbox)
print(f"\n📦 Caricamento backbone GLiNER: {MODEL_NAME}")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Freeze per sicurezza (Inference mode)
txt_enc.eval().to(DEVICE)
lbl_enc.eval().to(DEVICE)
proj.eval().to(DEVICE)

# 3. Caricamento Prompt Encoder (Trained)
checkpoint_path = select_checkpoint_interactive(SAVINGS_DIR)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Recuperiamo dimensioni dal backbone
vocab_size = lbl_enc.embeddings.word_embeddings.num_embeddings
embed_dim = lbl_enc.embeddings.word_embeddings.embedding_dim

prompt_encoder = PromptEncoder(vocab_size, embed_dim).to(DEVICE)
prompt_encoder.load_state_dict(checkpoint) # Carica solo i pesi del prompt encoder
prompt_encoder.eval()

print("✅ Prompt Encoder caricato e collegato.")

# 4. Pre-calcolo Matrice Label
label_matrix = build_optimized_label_matrix(
    prompt_encoder, lbl_enc, proj, lbl_tok, label_names, label2desc
)
print(f"✅ Matrice Label Pronta: {label_matrix.shape}")

# ==========================================================
# VALUTAZIONE LOOP
# ==========================================================
print(f"\n🔍 Valutazione su {len(test_records)} record...")

y_true_all = []
y_pred_all = []
n_skipped = 0

with torch.no_grad():
    for idx, record in enumerate(tqdm(test_records)):
        tokens = record["tokens"]
        labels = record["labels"]
        
        if len(tokens) != len(labels):
            n_skipped += 1
            continue
        
        # Tokenizzazione Testo
        input_ids = txt_tok.convert_tokens_to_ids(tokens)
        
        # Gestione lunghezza massima
        max_len = getattr(txt_tok, "model_max_length", 512)
        if len(input_ids) > max_len:
            input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
            labels = labels[:len(input_ids)]
        
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.long, device=DEVICE)
        
        # Encode Testo
        out_txt = txt_enc(input_ids=input_ids_tensor, attention_mask=attention_mask)
        H = F.normalize(out_txt.last_hidden_state, dim=-1) # [1, seq, 768]
        
        # Calcolo similarità con la Matrice Label ottimizzata
        logits = torch.matmul(H, label_matrix.T).squeeze(0) # [seq, num_labels]
        preds = logits.argmax(-1).cpu().numpy()
        
        # Raccolta risultati
        for pred, true_label in zip(preds, labels):
            if true_label != -100:
                y_true_all.append(true_label)
                y_pred_all.append(pred)

# ==========================================================
# REPORTING & EXPORT
# ==========================================================
all_label_ids = list(range(num_labels))

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="macro", zero_division=0, labels=all_label_ids
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average="micro", zero_division=0, labels=all_label_ids
)

class_report = classification_report(
    y_true_all, y_pred_all, target_names=label_names, labels=all_label_ids, zero_division=0, digits=4
)

print(f"\n{'='*70}")
print(f"📊 RISULTATI TEST (PROMPT TUNING)")
print(f"{'='*70}")
print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
print(f"Token valutati: {len(y_true_all):,}")
print(f"\n🎯 METRICHE:")
print(f"  Macro F1: {f1_macro:.4f}")
print(f"  Micro F1: {f1_micro:.4f}")
print(f"\n📋 REPORT:\n{class_report}")

# Export Markdown
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("test_results", exist_ok=True)
filename = f"test_results/eval_prompt_{timestamp}.md"

pred_counts = Counter(y_pred_all)
true_counts = Counter(y_true_all)

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Risultati Test - Prompt Tuning\n\n")
    f.write(f"**Checkpoint:** {os.path.basename(checkpoint_path)}\n")
    f.write(f"**Dataset:** {TEST_PATH}\n\n")
    f.write(f"## Metriche\n- **Macro F1:** {f1_macro:.4f}\n- **Micro F1:** {f1_micro:.4f}\n\n")
    f.write(f"## Report Classi\n```\n{class_report}\n```\n")
    f.write(f"## Distribuzione\n| Label | Pred | True | Diff |\n|---|---|---|---|\n")
    for lid in sorted(pred_counts.keys(), key=lambda x: pred_counts[x], reverse=True):
        f.write(f"| {id2label[lid]} | {pred_counts[lid]} | {true_counts[lid]} | {pred_counts[lid]-true_counts[lid]:+d} |\n")

print(f"\n💾 Report salvato: {filename}")