"""
Dataset: dataset_tokenlevel_balanced.json
Label set: derivato da label2desc.json / label2id.json
"""

import json, torch, torch.nn.functional as F
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm
import os

#layer in più per descrizioni, sblocca encoder input, dopo hard prompting
# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.9
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42

DATASET_PATH = "dataset/dataset_tokenlevel_balanced_3000.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

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

# ✨ DataLoader per tutto il dataset
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
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
        tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"),
        lbl_enc, proj, txt_enc, label2desc, optimizer, scheduler
    )
    
    epoch_time = time.time() - epoch_start_time
    
    # ✨ Traccia miglior epoca
    if train_loss < best_loss:
        best_loss = train_loss
        best_epoch = epoch
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} acc={train_acc:.1f}% | "
          f"lr={current_lr:.2e} | time={epoch_time:.1f}s")
    
    # ✨ Early stopping check
    early_stopping(train_loss)
    if early_stopping.early_stop:
        print(f"\n🛑 Early stopping triggered at epoch {epoch}")
        print(f"🏆 Best loss: {best_loss:.4f} (epoch {best_epoch})")
        break

# ✨ Timer totale
total_training_time = time.time() - total_start_time

print(f"\n⏱️  TEMPO TOTALE: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
print(f"🏆 Best loss: {best_loss:.4f} (epoch {best_epoch})")

# ==========================================================
# 💾 SALVATAGGIO MODELLO
# ==========================================================
print(f"\n💾 Salvataggio modello...")

os.makedirs("savings", exist_ok=True)
save_path = f"savings/model_bs{BATCH_SIZE}_ep{EPOCHS}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_temp{TEMPERATURE}_gc{GRAD_CLIP}_warmup{WARMUP_STEPS}_patience{EARLY_STOPPING_PATIENCE}.pt"

torch.save({
    'label_encoder_state_dict': lbl_enc.state_dict(),
    'projection_state_dict': proj.state_dict(),
}, save_path)

print(f"✅ Modello salvato: {save_path}")