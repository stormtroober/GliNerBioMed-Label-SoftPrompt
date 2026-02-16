# -*- coding: utf-8 -*-
"""
Training con OPTUNA - Ottimizzazione Iperparametri
- 20 trial
- Parametri: LR_EMBED, LR_PROJ, TEMPERATURE, GAMMA, BETA
- Metrica: Validation Loss (minimize)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json
import torch
import torch.nn.functional as F
import time
from datetime import datetime
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

# Per visualizzazioni
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ==========================================================
# 🔧 CONFIGURAZIONE BASE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
EARLY_STOPPING_PATIENCE = 4
RANDOM_SEED = 42

# OPTUNA SETTINGS
N_TRIALS = 20
STUDY_NAME = "gliner_soft_optimization"

# ==========================================
# PATHS
# ==========================================
if is_running_on_kaggle():
    path = "/kaggle/input/bc5dr-full/" 
    MODEL_NAME = "/kaggle/input/glinerbismall2/"
else:
    path = "../dataset/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

DATASET_PATH = path + "dataset_tknlvl_bi.json"
VAL_DATASET_PATH = path + "val_dataset_tknlvl_bi.json"
TEST_DATASET_PATH = path + "test_dataset_tknlvl_bi.json"
LABEL2DESC_PATH = path + "label2desc.json"
LABEL2ID_PATH = path + "label2id.json"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# FOCAL LOSS & CLASS BALANCED WEIGHTS
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            valid_mask = targets != self.ignore_index
            alpha_t = torch.ones_like(targets, dtype=logits.dtype)
            alpha_tensor = self.alpha.to(targets.device)
            alpha_t[valid_mask] = alpha_tensor[targets[valid_mask]]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        mask = targets != self.ignore_index
        if mask.sum() == 0: 
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return focal_loss[mask].mean()

def get_cb_weights(dataset_path, label2id, beta=0.9999):
    """Calcola i pesi Class Balanced"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    label_counts = Counter()
    for record in data:
        for label_id in record["labels"]:
            if label_id != -100:
                label_counts[label_id] += 1
                
    num_classes = len(label2id)
    weights = torch.ones(num_classes)
    
    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[label_id] = 1.0 / effective_num
        else:
            weights[label_id] = 0.0 
        
    weights = weights / weights.sum() * num_classes
    return weights

# ==========================================================
# DATASET & VALIDATION
# ==========================================================
def truncate_tokens_safe(tokens, tokenizer, max_len=None):
    if max_len is None: 
        max_len = tokenizer.model_max_length
    if len(tokens) <= max_len: 
        return tokens
    if tokens[0] == tokenizer.cls_token and tokens[-1] == tokenizer.sep_token and max_len >= 2:
        return [tokens[0]] + tokens[1:max_len-1] + [tokens[-1]]
    return tokens[:max_len]

def validate(val_records, txt_enc, txt_tok, soft_table, proj, device, criterion, temperature):
    """Validation con Soft Table - calcola loss e F1"""
    txt_enc.eval()
    soft_table.eval()
    proj.eval()
    
    y_true_all = []
    y_pred_all = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        for record in val_records:
            tokens = record["tokens"]
            labels = record["labels"]
            
            if len(tokens) != len(labels): 
                continue
            
            input_ids = txt_tok.convert_tokens_to_ids(tokens)
            
            max_len = getattr(txt_tok, "model_max_length", 512)
            if len(input_ids) > max_len:
                input_ids = truncate_tokens_safe(input_ids, txt_tok, max_len)
                labels = labels[:len(input_ids)]
            
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            
            out_txt = txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            logits = torch.matmul(H, label_matrix.T) / temperature
            
            # Calcola loss
            loss = criterion(logits.view(-1, label_matrix.size(0)), labels_tensor.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
            preds = logits.squeeze(0).argmax(-1).cpu().numpy()
            
            for pred, true_label in zip(preds, labels):
                if true_label != -100:
                    y_true_all.append(true_label)
                    y_pred_all.append(pred)
                    
    macro_f1 = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )[2]
    micro_f1 = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="micro", zero_division=0
    )[2]
    
    avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_val_loss, macro_f1, micro_f1

class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer):
        with open(path_json, "r", encoding="utf-8") as f: 
            self.records = json.load(f)
        self.tok = tokenizer
    
    def __len__(self): 
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(rec["labels"]),
        }

def collate_batch(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attn_mask, 
        "labels": labels
    }

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float('inf')
        self.early_stop = False
    
    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience: 
                self.early_stop = True
            return False

# ==========================================================
# CARICAMENTO DATI (UNA VOLTA SOLA)
# ==========================================================
print("📦 Caricamento dati e modello base...")

# Modello
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc_base = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj_base = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc_base.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Freeze encoders
for p in txt_enc_base.parameters(): 
    p.requires_grad = False
for p in lbl_enc.parameters(): 
    p.requires_grad = False

txt_enc_base.to(DEVICE)
lbl_enc.to(DEVICE)

# Labels
with open(LABEL2DESC_PATH) as f: 
    label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: 
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
label_names = [id2label[i] for i in range(num_labels)]

# Hard Embeddings
print("⚙️  Generazione Hard Embeddings...")
desc_texts = [label2desc[name] for name in label_names]
batch_desc = lbl_tok(
    desc_texts, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
).to(DEVICE)

with torch.no_grad():
    out = lbl_enc(**batch_desc)
    mask = batch_desc["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

HE_VECTORS = pooled
print(f"✅ Hard Embeddings: {HE_VECTORS.shape}")

# Dataset
print(f"📥 Caricamento dataset...")
ds = TokenJsonDataset(DATASET_PATH, txt_tok)

with open(VAL_DATASET_PATH, "r", encoding="utf-8") as f:
    val_records = json.load(f)
print(f"✅ Training: {len(ds)} | Validation: {len(val_records)}")

# ==========================================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def objective(trial):
    """Funzione obiettivo per Optuna - ritorna Validation Loss"""
    
    # ========== SAMPLING IPERPARAMETRI ==========
    lr_embed = trial.suggest_float("lr_embed", 1e-5, 1e-3, log=True)
    lr_proj = trial.suggest_float("lr_proj", 1e-5, 5e-4, log=True)
    temperature = trial.suggest_float("temperature", 0.1, 0.5)
    gamma = trial.suggest_float("gamma", 2.0, 6.0)
    beta = trial.suggest_categorical("beta", [0.999, 0.9999])
    
    print(f"\n🔬 TRIAL {trial.number + 1} | LR_E:{lr_embed:.6f} LR_P:{lr_proj:.6f} T:{temperature:.3f} γ:{gamma:.2f} β:{beta}")
    
    # ========== PREPARAZIONE MODELLO ==========
    txt_enc = txt_enc_base
    
    proj = nn.Linear(
        proj_base.in_features, 
        proj_base.out_features, 
        bias=proj_base.bias is not None
    )
    with torch.no_grad():
        proj.weight.copy_(proj_base.weight)
        if proj.bias is not None:
            proj.bias.copy_(proj_base.bias)
    proj.to(DEVICE)
    
    soft_table = nn.Embedding(num_labels, HE_VECTORS.size(1))
    with torch.no_grad():
        soft_table.weight.copy_(HE_VECTORS)
    soft_table.to(DEVICE)
    
    # ========== SETUP TRAINING ==========
    cb_weights = get_cb_weights(DATASET_PATH, label2id, beta=beta).to(DEVICE)
    criterion = FocalLoss(alpha=cb_weights, gamma=gamma, ignore_index=-100)
    
    optimizer = optim.AdamW([
        {"params": soft_table.parameters(), "lr": lr_embed},
        {"params": proj.parameters(), "lr": lr_proj}
    ], weight_decay=WEIGHT_DECAY)
    
    train_loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_batch(b, txt_tok.pad_token_id)
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=len(train_loader) * EPOCHS
    )
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # ========== TRAINING LOOP ==========
    txt_enc.eval()
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        soft_table.train()
        proj.train()
        
        # Inizio epoca
        print(f"  Epoch {epoch}/{EPOCHS} - Training...", end="", flush=True)
        
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with torch.no_grad():
                out_txt = txt_enc(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"]
                )
                H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
            
            label_vecs = proj(soft_table.weight)
            label_matrix = F.normalize(label_vecs, dim=-1)
            
            logits = torch.matmul(H_text, label_matrix.T) / temperature
            loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(soft_table.parameters()) + list(proj.parameters()), 
                GRAD_CLIP
            )
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # VALIDATION
        val_loss, val_macro_f1, val_micro_f1 = validate(
            val_records, txt_enc, txt_tok, soft_table, proj, DEVICE, criterion, temperature
        )
        
        # Fine epoca
        print(f" Loss:{avg_loss:.4f} Val_Loss:{val_loss:.4f} Val_F1_Macro:{val_macro_f1:.4f} Val_F1_Micro:{val_micro_f1:.4f}")
        
        # Report intermediate value per pruning (usa validation loss)
        trial.report(val_loss, epoch)
        
        # Pruning: ferma trial non promettenti
        if trial.should_prune():
            print(f"  ✂️  Pruned at epoch {epoch}")
            raise optuna.TrialPruned()
        
        # Track best (minimizzazione)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Early stopping (basato su loss - minimizzazione)
        # Invertiamo il segno per usare la stessa logica di EarlyStopping
        is_improved = early_stopping(-val_loss)
        if early_stopping.early_stop:
            print(f"  🛑 Early stopping")
            break
    
    print(f"  ✅ Completed | Best Val Loss: {best_val_loss:.4f}")
    
    # Cleanup memory
    del soft_table, proj, optimizer, scheduler, train_loader
    torch.cuda.empty_cache()
    
    return best_val_loss

# ==========================================================
# OPTUNA STUDY
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 INIZIO OTTIMIZZAZIONE OPTUNA")
    print("="*70)
    print(f"  Trials: {N_TRIALS} | Metric: Validation Loss (minimize) | Epochs: {EPOCHS}")
    print("="*70 + "\n")
    
    # Crea study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3
        ),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    # Ottimizzazione
    study.optimize(
        objective, 
        n_trials=N_TRIALS,
        show_progress_bar=False
    )
    
    # ==========================================================
    # RISULTATI
    # ==========================================================
    print("\n" + "="*70)
    print("🏆 OTTIMIZZAZIONE COMPLETATA")
    print("="*70)
    
    print(f"\n📊 STATISTICHE:")
    print(f"  Trials completati:  {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"  Trials pruned:      {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"  Trials failed:      {len([t for t in study.trials if t.state == TrialState.FAIL])}")
    
    print(f"\n🥇 BEST TRIAL:")
    best_trial = study.best_trial
    print(f"  Trial number:       {best_trial.number}")
    print(f"  Validation Loss: {best_trial.value:.4f}")
    print(f"\n  Parametri:")
    for key, value in best_trial.params.items():
        if 'lr' in key:
            print(f"    {key:12s}: {value:.6f}")
        else:
            print(f"    {key:12s}: {value}")
    
    # Top 5 trials (ordinati per loss crescente - migliori = loss più bassa)
    print(f"\n📈 TOP 5 TRIALS (Best = Lowest Loss):")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    for i, trial in enumerate(top_trials, 1):
        if trial.value is not None:
            print(f"  {i}. Trial {trial.number:2d} | Loss: {trial.value:.4f} | "
                  f"LR_Embed: {trial.params['lr_embed']:.6f} | "
                  f"Temp: {trial.params['temperature']:.3f} | "
                  f"Gamma: {trial.params['gamma']:.2f}")
    
    # ==========================================================
    # SALVATAGGIO RISULTATI
    # ==========================================================
    os.makedirs("optuna_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salva study object
    study_path = f"optuna_results/study_{timestamp}.pkl"
    import joblib
    joblib.dump(study, study_path)
    print(f"\n💾 Study salvato in: {study_path}")
    
    # Salva report dettagliato
    report_path = f"optuna_results/optuna_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Optuna Optimization Report\n\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Environment:** {'Kaggle' if is_running_on_kaggle() else 'Local'}\n")
        f.write(f"**Study Name:** {STUDY_NAME}\n\n")
        
        f.write(f"## Configuration\n")
        f.write(f"- **Trials:** {N_TRIALS}\n")
        f.write(f"- **Metric:** Validation Loss (minimize)\n")
        f.write(f"- **Epochs per trial:** {EPOCHS}\n")
        f.write(f"- **Early stopping patience:** {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"- **Pruner:** MedianPruner\n\n")
        
        f.write(f"## Search Space\n")
        f.write(f"- **lr_embed:** [1e-5, 1e-3] (log scale)\n")
        f.write(f"- **lr_proj:** [1e-5, 5e-4] (log scale)\n")
        f.write(f"- **temperature:** [0.1, 0.5]\n")
        f.write(f"- **gamma:** [2.0, 6.0]\n")
        f.write(f"- **beta:** {0.999, 0.9999}\n\n")
        
        f.write(f"## Results Summary\n")
        f.write(f"- **Trials completed:** {len([t for t in study.trials if t.state == TrialState.COMPLETE])}\n")
        f.write(f"- **Trials pruned:** {len([t for t in study.trials if t.state == TrialState.PRUNED])}\n")
        f.write(f"- **Best Value:** {study.best_value:.4f}\n\n")
        
        f.write(f"## Best Parameters\n")
        f.write(f"```json\n")
        f.write(json.dumps(best_trial.params, indent=2))
        f.write(f"\n```\n\n")
        
        f.write(f"## Top 10 Trials\n")
        f.write(f"| Rank | Trial | Val Loss | LR_Embed | LR_Proj | Temp | Gamma | Beta |\n")
        f.write(f"|------|-------|----------|----------|---------|------|-------|------|\n")
        
        top_10 = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:10]
        for i, trial in enumerate(top_10, 1):
            if trial.value is not None:
                f.write(f"| {i} | {trial.number} | {trial.value:.4f} | "
                       f"{trial.params['lr_embed']:.6f} | "
                       f"{trial.params['lr_proj']:.6f} | "
                       f"{trial.params['temperature']:.3f} | "
                       f"{trial.params['gamma']:.2f} | "
                       f"{trial.params['beta']} |\n")
        
        f.write(f"\n## All Trials Details\n")
        for trial in study.trials:
            f.write(f"\n### Trial {trial.number}\n")
            f.write(f"- **State:** {trial.state}\n")
            if trial.value is not None:
                f.write(f"- **Value:** {trial.value:.4f}\n")
            if trial.params:
                f.write(f"- **Params:**\n")
                for key, value in trial.params.items():
                    f.write(f"  - {key}: {value}\n")
    
    print(f"💾 Report salvato in: {report_path}")
    
    # Salva JSON
    results_json = {
        "timestamp": timestamp,
        "n_trials": N_TRIALS,
        "best_value": study.best_value,
        "best_params": best_trial.params,
        "best_trial_number": best_trial.number,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state)
            }
            for t in study.trials
        ]
    }
    
    json_path = f"optuna_results/results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"💾 JSON salvato in: {json_path}")