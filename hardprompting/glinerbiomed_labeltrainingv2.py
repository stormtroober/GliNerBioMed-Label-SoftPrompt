"""
Dataset: dataset_tokenlevel_balanced.json
Label set: derivato da label2desc.json / label2id.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json, torch, torch.nn.functional as F
import time
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

#layer in più per descrizioni, sblocca encoder input, dopo hard prompting
# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 3.272473169674807e-05
WEIGHT_DECAY = 7.430485402967551e-05
TEMPERATURE = 0.5113393592758555
GRAD_CLIP = 1.1884977591347257
WARMUP_STEPS = 59
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42
VAL_SPLIT_RATIO = 0.2  # Usato solo se USE_SEPARATE_VAL_FILE = False

# 🔄 Flag per gestione validation:
# - True: usa file di validation separato (es. dataset_anatEM ha val_dataset_tknlvl_bi.json)
# - False: split del training set (80% train, 20% val)
USE_SEPARATE_VAL_FILE = False

# ==========================================
# KAGGLE / LOCAL PATHS
# ==========================================
if is_running_on_kaggle():
    path = "/kaggle/input/bc5dr-full/"
    #path = "/kaggle/input/tknlvl-jnlpa-5k/"
    MODEL_NAME = "/kaggle/input/glinerbismall2/"
else:
    path = "../dataset/"
    path_val = "../dataset_anatEM/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

DATASET_PATH = path + "dataset_tknlvl_bi.json"
LABEL2DESC_PATH = path + "label2desc.json"
LABEL2ID_PATH = path + "label2id.json"

# Path validation (usato solo se USE_SEPARATE_VAL_FILE = True)
if USE_SEPARATE_VAL_FILE:
    if is_running_on_kaggle():
        VAL_DATASET_PATH = "/kaggle/input/bc5dr-full/val_dataset_tknlvl_bi.json"
    else:
        VAL_DATASET_PATH = "../dataset_anatEM/val_dataset_tknlvl_bi.json"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 0️⃣ MODELLO + TOKENIZER
# ==========================================================
print("📦 Caricamento modello base GLiNER-BioMed...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Congeliamo il text encoder
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = True
for p in proj.parameters(): p.requires_grad = True

# ==========================================================
# 1️⃣ LABELS E DESCRIZIONI
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())

def compute_label_matrix(label2desc: dict, lbl_enc, lbl_tok, proj) -> torch.Tensor:
    """Embedda le descrizioni con lbl_enc + proj (trainabili)."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.set_grad_enabled(lbl_enc.training):
        out = lbl_enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)   # [num_labels, hidden_dim]

# ==========================================================
# 2️⃣ DATASET TOKEN-LEVEL
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, label2id):
        with open(path_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]

        input_ids = self.tok.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        y = []
        for lab in labels:
            if lab == -100: y.append(-100)
            else: y.append(lab)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(y),
        }

def collate_batch(batch, pad_id, ignore_index=-100):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), ignore_index, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

# ==========================================================
# 3️⃣ TRAINING SETUP
# ==========================================================
print("📚 Caricamento dataset...")
ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)

print(f"📊 Total dataset size: {len(ds)}\n")

def compute_class_weights(data_path, label2id):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    counts = torch.zeros(len(label2id))
    total = 0
    
    for record in data:
        for label in record["labels"]:
            if label != -100:
                counts[label] += 1
                total += 1
    
    weights = total / (len(label2id) * counts.clamp(min=1))
    print(f"🔧 Class weights: {weights}")
    return weights

class_weights = compute_class_weights(DATASET_PATH, label2id).to(DEVICE)
ce_loss = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

# ✨ Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# ✨ Funzione per eseguire un'epoca di training
def run_epoch(loader, lbl_enc, proj, txt_enc, label2desc, optimizer, scheduler=None):
    """Esegue un'epoca di training"""
    lbl_enc.train()
    proj.train()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            out_txt = txt_enc(**{k: batch[k] for k in ["input_ids","attention_mask"]})
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        label_matrix = compute_label_matrix(label2desc, lbl_enc, lbl_tok, proj)
        logits = torch.matmul(H, label_matrix.T) / TEMPERATURE
        loss = ce_loss(logits.view(-1, len(label_names)), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lbl_enc.parameters()) + list(proj.parameters()), GRAD_CLIP)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        total_acc += (preds[mask] == batch["labels"][mask]).float().sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = (total_acc / n_tokens) * 100
    return avg_loss, avg_acc

# ✨ Funzione per eseguire validation (senza gradient)
@torch.no_grad()
def run_validation_epoch(loader, lbl_enc, proj, txt_enc, label2desc):
    """Esegue validation (no gradient)"""
    lbl_enc.eval()
    proj.eval()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        out_txt = txt_enc(**{k: batch[k] for k in ["input_ids","attention_mask"]})
        H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        label_matrix = compute_label_matrix(label2desc, lbl_enc, lbl_tok, proj)
        logits = torch.matmul(H, label_matrix.T) / TEMPERATURE
        loss = ce_loss(logits.view(-1, len(label_names)), batch["labels"].view(-1))
        
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        total_acc += (preds[mask] == batch["labels"][mask]).float().sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = (total_acc / n_tokens) * 100
    return avg_loss, avg_acc

# ✨ Timer totale
total_start_time = time.time()

# ==========================================================
# 4️⃣ TRAINING LOOP
# ==========================================================
print("\n🚀 Inizio training...\n")

# Carica modelli su device
txt_enc.eval().to(DEVICE)
lbl_enc.train().to(DEVICE)
proj.train().to(DEVICE)

# ✨ DataLoader per training e validation
if USE_SEPARATE_VAL_FILE:
    # Usa file di validation separato
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    
    print("📚 Caricamento validation dataset da file separato...")
    val_ds = TokenJsonDataset(VAL_DATASET_PATH, txt_tok, label2id)
    print(f"📊 Validation dataset size: {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
else:
    # Split del training set
    print(f"📚 Splitting training set ({int((1-VAL_SPLIT_RATIO)*100)}% train, {int(VAL_SPLIT_RATIO*100)}% val)...")
    val_size = int(len(ds) * VAL_SPLIT_RATIO)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    print(f"📊 Training size: {len(train_ds)}, Validation size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))

optimizer = optim.Adam(list(lbl_enc.parameters()) + list(proj.parameters()), 
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
)

training_start_time = time.time()

# ✨ Inizializza early stopping
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
best_loss = float('inf')
best_epoch = 0

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    # Training
    train_loss, train_acc = run_epoch(
        tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"),
        lbl_enc, proj, txt_enc, label2desc, optimizer, scheduler
    )
    
    # ✨ Validation
    val_loss, val_acc = run_validation_epoch(
        tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False),
        lbl_enc, proj, txt_enc, label2desc
    )
    
    epoch_time = time.time() - epoch_start_time
    
    # ✨ Traccia miglior epoca basata su validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | lr={current_lr:.2e} | time={epoch_time:.1f}s")
    
    # ✨ Early stopping check basato su validation loss
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\n🛑 Early stopping triggered at epoch {epoch}")
        print(f"🏆 Best val_loss: {best_loss:.4f} (epoch {best_epoch})")
        break

# ✨ Timer totale
total_training_time = time.time() - total_start_time

print(f"\n⏱️  TEMPO TOTALE: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
print(f"🏆 Best val_loss: {best_loss:.4f} (epoch {best_epoch})")

# ==========================================================
# 💾 SALVATAGGIO MODELLO
# ==========================================================
print(f"\n💾 Salvataggio modello...")

os.makedirs("savings", exist_ok=True)

# Estrai nome dataset dal path
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
dataset_size = len(ds)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Nome file con dataset info e timestamp (parametri salvati dentro il .pt)
save_path = f"savings/model_{dataset_name}_size{dataset_size}_{timestamp}.pt"

torch.save({
    # State dicts
    'label_encoder_state_dict': lbl_enc.state_dict(),
    'projection_state_dict': proj.state_dict(),
    
    # Dataset info
    'dataset_name': dataset_name,
    'dataset_path': DATASET_PATH,
    'dataset_size': dataset_size,
    'use_separate_val_file': USE_SEPARATE_VAL_FILE,
    'val_dataset_path': VAL_DATASET_PATH if USE_SEPARATE_VAL_FILE else None,
    'val_split_ratio': None if USE_SEPARATE_VAL_FILE else VAL_SPLIT_RATIO,
    'val_dataset_size': len(val_ds),
    
    # Tutti gli iperparametri
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'temperature': TEMPERATURE,
        'grad_clip': GRAD_CLIP,
        'warmup_steps': WARMUP_STEPS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'random_seed': RANDOM_SEED,
    },
    
    # Training info
    'training_info': {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'total_training_time_seconds': total_training_time,
        'model_name': MODEL_NAME,
    },
    
    # Label mappings (utili per inference)
    'label2id': label2id,
    'label2desc': label2desc,
}, save_path)

print(f"✅ Modello salvato: {save_path}")
print(f"📊 Dataset: {dataset_name} (size: {dataset_size})")
print(f"📋 Parametri salvati nel checkpoint")