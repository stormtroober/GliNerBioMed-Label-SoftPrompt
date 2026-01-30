"""
🎯 Optuna Hyperparameter Optimization per GLiNER-BioMed Label Training
Dataset: dataset_tokenlevel_balanced.json
Ottimizzazione: Learning Rate, Weight Decay, Temperature, Batch Size, Warmup Steps, Grad Clip
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json, torch, torch.nn.functional as F
import time
import optuna
from optuna.trial import TrialState
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
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
N_TRIALS = 25  # Numero di trial (tra 20 e 30)
STUDY_NAME = "gliner_biomed_hpo"
PRUNER_WARMUP = 3  # Epoche minime prima di pruning

# ==========================================
# KAGGLE / LOCAL PATHS
# ==========================================
if is_running_on_kaggle():
    path = "/kaggle/input/anatem_tknlvl_bi_train/"
    MODEL_NAME = "/kaggle/input/glinerbismall2/"
else:
    path = "../dataset/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

DATASET_PATH = path + "dataset_tknlvl_bi.json"
LABEL2DESC_PATH = path + "label2desc.json"
LABEL2ID_PATH = path + "label2id.json"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 0️⃣ LABEL MAPPINGS (caricati una volta)
# ==========================================================
print("📚 Caricamento label mappings...")
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = list(label2desc.keys())

print(f"📊 Labels trovate: {len(label_names)}")

# ==========================================================
# 1️⃣ DATASET CLASS
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
# 2️⃣ UTILITY FUNCTIONS
# ==========================================================
def compute_class_weights(data_path, label2id):
    """Calcola pesi per bilanciare le classi"""
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
    return weights

def compute_label_matrix(label2desc, label_names, lbl_enc, lbl_tok, proj, device):
    """Embedda le descrizioni con lbl_enc + proj (trainabili)."""
    desc_texts = [label2desc[k] for k in label_names]
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.set_grad_enabled(lbl_enc.training):
        out = lbl_enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vecs = proj(pooled)
    return F.normalize(vecs, dim=-1)

def run_training_epoch(loader, lbl_enc, proj, txt_enc, label2desc, label_names, 
                       lbl_tok, optimizer, scheduler, ce_loss, temperature,
                       grad_clip, device):
    """Esegue un'epoca di training"""
    lbl_enc.train()
    proj.train()
    
    total_loss, total_acc, n_tokens = 0.0, 0.0, 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            out_txt = txt_enc(**{k: batch[k] for k in ["input_ids","attention_mask"]})
            H = F.normalize(out_txt.last_hidden_state, dim=-1)
        
        label_matrix = compute_label_matrix(label2desc, label_names, lbl_enc, lbl_tok, proj, device)
        logits = torch.matmul(H, label_matrix.T) / temperature
        loss = ce_loss(logits.view(-1, len(label_names)), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lbl_enc.parameters()) + list(proj.parameters()), grad_clip)
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

# ==========================================================
# 3️⃣ OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def objective(trial):
    """Funzione obiettivo per Optuna"""
    
    # 🎯 HYPERPARAMETERS DA OTTIMIZZARE
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    temperature = trial.suggest_float("temperature", 0.5, 1.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 20, 100)
    grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
    
    print(f"\n{'='*60}")
    print(f"🔬 Trial {trial.number}")
    print(f"   lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"   temp={temperature:.2f}, batch={batch_size}")
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
        
        # Freeze text encoder
        for p in txt_enc.parameters(): p.requires_grad = False
        for p in lbl_enc.parameters(): p.requires_grad = True
        for p in proj.parameters(): p.requires_grad = True
        
        txt_enc.eval().to(DEVICE)
        lbl_enc.train().to(DEVICE)
        proj.train().to(DEVICE)
        
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        raise optuna.TrialPruned()
    
    # Dataset e DataLoader
    ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id))
    
    # Class weights e loss
    class_weights = compute_class_weights(DATASET_PATH, label2id).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
    
    # Optimizer e Scheduler
    optimizer = optim.Adam(list(lbl_enc.parameters()) + list(proj.parameters()), 
                          lr=learning_rate, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_training_epoch(
            tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False),
            lbl_enc, proj, txt_enc, label2desc, label_names,
            lbl_tok, optimizer, scheduler, ce_loss, temperature,
            grad_clip, DEVICE
        )
        
        print(f"  Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        
        # Report per pruning
        trial.report(train_loss, epoch)
        
        # Pruning check (dopo warmup)
        if trial.should_prune() and epoch >= PRUNER_WARMUP:
            # Cleanup
            del model, txt_enc, lbl_enc, proj, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        
        # Early stopping interno
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  ⚠️ Early stopping at epoch {epoch}")
                break
    
    # Cleanup
    del model, txt_enc, lbl_enc, proj, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✅ Trial {trial.number} completato: best_loss={best_loss:.4f}")
    
    return best_loss

# ==========================================================
# 4️⃣ OPTUNA CALLBACKS
# ==========================================================
def print_callback(study, trial):
    """Callback per stampare progresso"""
    if trial.state == TrialState.COMPLETE:
        print(f"\n📊 Trial {trial.number} completato: {trial.value:.4f}")
        print(f"   Migliore finora: {study.best_value:.4f} (trial {study.best_trial.number})")

# ==========================================================
# 5️⃣ MAIN: OPTUNA STUDY
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 OPTUNA HYPERPARAMETER OPTIMIZATION - GLiNER-BioMed")
    print("="*70)
    print(f"📍 Device: {DEVICE}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"🔄 Trials: {N_TRIALS}")
    print(f"📈 Epochs per trial: {EPOCHS}")
    print("="*70 + "\n")
    
    # Crea studio con MedianPruner per eliminare trial poco promettenti
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
        gc_after_trial=True  # Force garbage collection
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
        }
    }
    
    results_path = f"optuna_results/optuna_study_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Risultati salvati: {results_path}")
    
    # ==========================================================
    # 📋 CODICE CONFIGURAZIONE OTTIMALE
    # ==========================================================
    print("\n" + "="*70)
    print("📋 CONFIGURAZIONE OTTIMALE (copia in glinerbiomed_labeltrainingv2.py)")
    print("="*70)
    print(f"""
# 🎯 Iperparametri ottimizzati da Optuna (Trial #{best_trial.number})
BATCH_SIZE = {best_trial.params['batch_size']}
LEARNING_RATE = {best_trial.params['learning_rate']:.6e}
WEIGHT_DECAY = {best_trial.params['weight_decay']:.6e}
TEMPERATURE = {best_trial.params['temperature']:.4f}
GRAD_CLIP = {best_trial.params['grad_clip']:.4f}
WARMUP_STEPS = {best_trial.params['warmup_steps']}
""")
    print("="*70)
    
    # ==========================================================
    # 📊 VISUALIZZAZIONE (opzionale)
    # ==========================================================
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # History plot
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        
        axes[0].scatter(trial_numbers, trial_values, alpha=0.7)
        axes[0].plot(trial_numbers, [min(trial_values[:i+1]) for i in range(len(trial_values))], 
                     'r-', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Loss')
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
        plot_path = f"optuna_results/optuna_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n📈 Plot salvato: {plot_path}")
        plt.close()
        
    except ImportError:
        print("\n⚠️ matplotlib non disponibile, skip plot")
    except Exception as e:
        print(f"\n⚠️ Errore plot: {e}")
    
    print("\n✅ Ottimizzazione completata!")
