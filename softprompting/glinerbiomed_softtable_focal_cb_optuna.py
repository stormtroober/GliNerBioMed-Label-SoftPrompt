# -*- coding: utf-8 -*-
"""
🎯 Optuna Hyperparameter Optimization per GLiNER-BioMed Soft Prompting
- Architettura: Soft Embedding Table + Projection (NO MLP)
- Strategia Loss: Class Balanced Weights + Focal Loss
- Ottimizzazione: LR Embed, LR Proj, Gamma Focal, CB Beta, Temperature, Batch Size, etc.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json
import torch
import torch.nn.functional as F
import time
import numpy as np
import optuna
from optuna.trial import TrialState
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm
import gc

# ==========================================================
# 🔧 CONFIGURAZIONE BASE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
EPOCHS = 10  # Ridotto per velocizzare trial (Optuna usa early stopping)
EARLY_STOPPING_PATIENCE = 3

# Configurazione Optuna
N_TRIALS = 25  # Numero di trial
STUDY_NAME = "gliner_biomed_softprompt_hpo"
PRUNER_WARMUP = 3  # Epoche minime prima di pruning
VAL_SPLIT_RATIO = 0.2  # Usato solo se USE_SEPARATE_VAL_FILE = False

# 🔄 Flag per gestione validation:
# - True: usa file di validation separato
# - False: split del training set (80% train, 20% val)
USE_SEPARATE_VAL_FILE = True

# ==========================================
# KAGGLE / LOCAL PATHS
# ==========================================
if is_running_on_kaggle():
    path = "/kaggle/input/bc5dr-full/"
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
# 0️⃣ LABEL MAPPINGS (caricati una volta)
# ==========================================================
print("📚 Caricamento label mappings...")
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())
num_labels = len(label2id)

print(f"📊 Labels trovate: {num_labels}")

# ==========================================================
# 1️⃣ FOCAL LOSS & CLASS BALANCED WEIGHTS
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
        
        # Calcolo base Focal
        focal_weight = (1 - p_t) ** self.gamma
        
        # Applicazione Alpha (Class Balanced Weights)
        if self.alpha is not None:
            valid_mask = targets != self.ignore_index
            alpha_t = torch.ones_like(targets, dtype=logits.dtype)
            alpha_tensor = self.alpha.to(targets.device)
            alpha_t[valid_mask] = alpha_tensor[targets[valid_mask]]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        # Ritorna la media sui token validi
        mask = targets != self.ignore_index
        if mask.sum() == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return focal_loss[mask].mean()

def get_cb_weights(dataset_path, label2id, beta=0.9999):
    """Calcola i pesi 'Class Balanced' (Cui et al. 2019)"""
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
        
    # Normalizzazione: somma pesi = numero classi
    weights = weights / weights.sum() * num_classes
    return weights

# ==========================================================
# 2️⃣ DATASET CLASS
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, label2id=None):
        with open(path_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1] * len(input_ids)),
            "labels": torch.tensor(rec["labels"]),
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
# 3️⃣ UTILITY FUNCTIONS
# ==========================================================
def generate_hard_embeddings(label2desc, label_names, lbl_enc, lbl_tok, device):
    """Genera Hard Embeddings dalle descrizioni"""
    desc_texts = [label2desc[name] for name in label_names]
    batch_desc = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        out = lbl_enc(**batch_desc)
        mask = batch_desc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled

def run_training_epoch(loader, soft_table, proj, txt_enc, criterion, 
                       optimizer, scheduler, temperature, grad_clip, device, num_labels):
    """Esegue un'epoca di training"""
    soft_table.train()
    proj.train()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 1. Text Embeddings (Frozen)
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        # 2. Label Embeddings (Soft Table diretta)
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        # 3. Similarità e Loss
        logits = torch.matmul(H_text, label_matrix.T) / temperature
        loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(soft_table.parameters()) + list(proj.parameters()), grad_clip)
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

@torch.no_grad()
def run_validation_epoch(loader, soft_table, proj, txt_enc, criterion, 
                         temperature, device, num_labels):
    """Esegue validation (no gradient)"""
    soft_table.eval()
    proj.eval()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        H_text = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        label_vecs = proj(soft_table.weight)
        label_matrix = F.normalize(label_vecs, dim=-1)
        
        logits = torch.matmul(H_text, label_matrix.T) / temperature
        loss = criterion(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        mask = batch["labels"] != -100
        preds = logits.argmax(-1)
        total_acc += (preds[mask] == batch["labels"][mask]).float().sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = (total_acc / n_tokens) * 100
    return avg_loss, avg_acc

# ==========================================================
# 4️⃣ OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def objective(trial):
    """Funzione obiettivo per Optuna - Soft Prompting"""
    
    # 🎯 HYPERPARAMETERS DA OTTIMIZZARE
    lr_embed = trial.suggest_float("lr_embed", 1e-4, 1e-2, log=True)
    lr_proj = trial.suggest_float("lr_proj", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    temperature = trial.suggest_float("temperature", 0.5, 2.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 20, 100)
    grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
    
    # Parametri specifici Focal Loss + CB
    gamma_focal = trial.suggest_float("gamma_focal", 1.0, 5.0)
    cb_beta = trial.suggest_float("cb_beta", 0.99, 0.9999)
    
    print(f"\n{'='*60}")
    print(f"🔬 Trial {trial.number}")
    print(f"   lr_embed={lr_embed:.2e}, lr_proj={lr_proj:.2e}, wd={weight_decay:.2e}")
    print(f"   temp={temperature:.2f}, batch={batch_size}")
    print(f"   gamma_focal={gamma_focal:.2f}, cb_beta={cb_beta:.4f}")
    print(f"   warmup={warmup_steps}, grad_clip={grad_clip:.2f}")
    print(f"{'='*60}")
    
    # Carica modello fresco per ogni trial
    try:
        model = GLiNER.from_pretrained(MODEL_NAME)
        core = model.model
        
        txt_enc = core.token_rep_layer.bert_layer.model
        lbl_enc = core.token_rep_layer.labels_encoder.model
        proj = core.token_rep_layer.labels_projection
        
        txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
        lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)
        
        # Freeze encoders
        for p in txt_enc.parameters(): p.requires_grad = False
        for p in lbl_enc.parameters(): p.requires_grad = False
        for p in proj.parameters(): p.requires_grad = True
        
        txt_enc.eval().to(DEVICE)
        lbl_enc.eval().to(DEVICE)
        proj.train().to(DEVICE)
        
        # Genera Hard Embeddings e crea Soft Table
        label_names_ordered = [id2label[i] for i in range(num_labels)]
        HE_VECTORS = generate_hard_embeddings(label2desc, label_names_ordered, lbl_enc, lbl_tok, DEVICE)
        
        soft_table = nn.Embedding(num_labels, HE_VECTORS.size(1))
        with torch.no_grad():
            soft_table.weight.copy_(HE_VECTORS)
        soft_table.train().to(DEVICE)
        
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        raise optuna.TrialPruned()
    
    # Dataset e DataLoader
    ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)
    
    # 🔄 Gestione validation in base al flag
    if USE_SEPARATE_VAL_FILE:
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
        
        val_ds = TokenJsonDataset(VAL_DATASET_PATH, txt_tok, label2id)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    else:
        val_size = int(len(ds) * VAL_SPLIT_RATIO)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], 
                                         generator=torch.Generator().manual_seed(RANDOM_SEED))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    
    # Class Balanced Weights e Focal Loss
    cb_weights = get_cb_weights(DATASET_PATH, label2id, beta=cb_beta).to(DEVICE)
    criterion = FocalLoss(alpha=cb_weights, gamma=gamma_focal, ignore_index=-100)
    
    # Optimizer con Learning Rates differenziati
    optimizer = optim.AdamW([
        {"params": soft_table.parameters(), "lr": lr_embed},
        {"params": proj.parameters(), "lr": lr_proj}
    ], weight_decay=weight_decay)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Training epoch
        train_loss, train_acc = run_training_epoch(
            tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS} [Train]", leave=False),
            soft_table, proj, txt_enc, criterion, optimizer, scheduler,
            temperature, grad_clip, DEVICE, num_labels
        )
        
        # Validation epoch
        val_loss, val_acc = run_validation_epoch(
            tqdm(val_loader, desc=f"  Epoch {epoch}/{EPOCHS} [Val]", leave=False),
            soft_table, proj, txt_enc, criterion,
            temperature, DEVICE, num_labels
        )
        
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}% | val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%")
        
        # Report per pruning
        trial.report(val_loss, epoch)
        
        # Pruning check (dopo warmup)
        if trial.should_prune() and epoch >= PRUNER_WARMUP:
            del model, txt_enc, lbl_enc, proj, soft_table, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        
        # Early stopping interno
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  ⚠️ Early stopping at epoch {epoch}")
                break
    
    # Cleanup
    del model, txt_enc, lbl_enc, proj, soft_table, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✅ Trial {trial.number} completato: best_val_loss={best_loss:.4f}")
    
    return best_loss

# ==========================================================
# 5️⃣ OPTUNA CALLBACKS
# ==========================================================
def print_callback(study, trial):
    """Callback per stampare progresso"""
    if trial.state == TrialState.COMPLETE:
        print(f"\n📊 Trial {trial.number} completato: {trial.value:.4f}")
        print(f"   Migliore finora: {study.best_value:.4f} (trial {study.best_trial.number})")

# ==========================================================
# 6️⃣ MAIN: OPTUNA STUDY
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 OPTUNA HPO - GLiNER-BioMed SOFT PROMPTING (Focal + CB)")
    print("="*70)
    print(f"📍 Device: {DEVICE}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"🔄 Trials: {N_TRIALS}")
    print(f"📈 Epochs per trial: {EPOCHS}")
    print("="*70 + "\n")
    
    # Crea studio con MedianPruner
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=PRUNER_WARMUP,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    start_time = time.time()
    
    # Esegui ottimizzazione
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        callbacks=[print_callback],
        gc_after_trial=True
    )
    
    total_time = time.time() - start_time
    
    # ==========================================================
    # 📊 RISULTATI
    # ==========================================================
    print("\n" + "="*70)
    print("🏆 RISULTATI OTTIMIZZAZIONE")
    print("="*70)
    
    # Statistiche trial
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"\n📈 Statistiche:")
    print(f"   - Trial completati: {len(completed_trials)}")
    print(f"   - Trial pruned: {len(pruned_trials)}")
    print(f"   - Tempo totale: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # Best trial
    best_trial = study.best_trial
    print(f"\n🥇 MIGLIOR TRIAL: #{best_trial.number}")
    print(f"   Loss: {best_trial.value:.4f}")
    print(f"\n   Iperparametri ottimali:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")
    
    # ==========================================================
    # 💾 SALVATAGGIO RISULTATI
    # ==========================================================
    os.makedirs("optuna_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salva studio completo
    results = {
        "study_name": STUDY_NAME,
        "timestamp": timestamp,
        "n_trials": N_TRIALS,
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "total_time_seconds": total_time,
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.state == TrialState.COMPLETE else None,
                "state": str(t.state),
                "params": t.params,
            }
            for t in study.trials
        ],
        "fixed_params": {
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "random_seed": RANDOM_SEED,
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "use_separate_val_file": USE_SEPARATE_VAL_FILE,
            "val_dataset_path": VAL_DATASET_PATH if USE_SEPARATE_VAL_FILE else None,
            "val_split_ratio": None if USE_SEPARATE_VAL_FILE else VAL_SPLIT_RATIO,
        }
    }
    
    results_path = f"optuna_results/softprompt_optuna_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Risultati salvati: {results_path}")
    
    # ==========================================================
    # 📋 CODICE CONFIGURAZIONE OTTIMALE
    # ==========================================================
    print("\n" + "="*70)
    print("📋 CONFIGURAZIONE OTTIMALE (copia in glinerbiomed_softtable_focal_cb.py)")
    print("="*70)
    print(f"""
# 🎯 Iperparametri ottimizzati da Optuna (Trial #{best_trial.number})
BATCH_SIZE = {best_trial.params['batch_size']}
LR_EMBED = {best_trial.params['lr_embed']:.6e}
LR_PROJ = {best_trial.params['lr_proj']:.6e}
WEIGHT_DECAY = {best_trial.params['weight_decay']:.6e}
TEMPERATURE = {best_trial.params['temperature']:.4f}
GRAD_CLIP = {best_trial.params['grad_clip']:.4f}
WARMUP_STEPS = {best_trial.params['warmup_steps']}

# Focal Loss + Class Balanced
GAMMA_FOCAL_LOSS = {best_trial.params['gamma_focal']:.4f}
CB_BETA = {best_trial.params['cb_beta']:.6f}
""")
    print("="*70)
    
    # ==========================================================
    # 📊 VISUALIZZAZIONE (opzionale)
    # ==========================================================
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # History plot
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        
        axes[0].scatter(trial_numbers, trial_values, alpha=0.7)
        axes[0].plot(trial_numbers, [min(trial_values[:i+1]) for i in range(len(trial_values))], 
                     'r-', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Parameter importance
        params_names = list(best_trial.params.keys())
        importance_values = []
        for p in params_names:
            try:
                imp = optuna.importance.get_param_importances(study)[p]
            except:
                imp = 0
            importance_values.append(imp)
        
        axes[1].barh(params_names, importance_values, color='steelblue')
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Hyperparameter Importance')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = f"optuna_results/softprompt_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n📈 Plot salvato: {plot_path}")
        plt.close()
        
    except ImportError:
        print("\n⚠️ matplotlib non disponibile, skip plot")
    except Exception as e:
        print(f"\n⚠️ Errore plot: {e}")
    
    print("\n✅ Ottimizzazione completata!")
