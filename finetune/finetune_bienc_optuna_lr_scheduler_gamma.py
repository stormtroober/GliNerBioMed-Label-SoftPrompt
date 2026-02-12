
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json 
import random
import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from gliner import GLiNER
import shutil
import optuna
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

# Dataset Configuration (choose one)
# For JNLPA dataset
DATASET_FOLDER = "dataset"  # or "dataset_bc5cdr" for BC5CDR dataset

if is_running_on_kaggle():
    path = "/kaggle/input/jnlpa15k/"
    train_path = path + "dataset_span_bi.json"
    val_path = path + "val_dataset_span_bi.json"
    test_path = path + "test_dataset_span_bi.json"
    label2id_path = path + "label2id.json"
else:
    # Local paths with configurable dataset folder
    train_path = f"{DATASET_FOLDER}/dataset_span_bi.json"
    val_path = f"{DATASET_FOLDER}/val_dataset_span_bi.json"
    test_path = f"{DATASET_FOLDER}/test_dataset_span_bi.json"
    label2id_path = f"{DATASET_FOLDER}/label2id.json"

# Training Configuration
target_steps = None       # Se impostato, il training si fermerà esattamente a questi step
target_epochs = 3         # Reduced to 3 for faster Optuna trials
batch_size = 32

# ⏱️ TIME ESTIMATION (based on 133.13s per epoch)
# - Time per trial (3 epochs): ~399.39s (~6.66 min)
# - Total time (15 trials): ~100 min (1h 40min)
# - With pruning: significantly reduced for unpromising trials

# ==========================================
# METRICS FUNCTION
# ==========================================
def calculate_metrics(dataset, model, batch_size=32):
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Gather all labels from dataset (already converted to short labels)
    all_labels = set()
    for d in dataset:
        for x in d['ner']: 
            all_labels.add(x[2])
    
    # Ensure 'O' is not in the label list
    if 'O' in all_labels:
        all_labels.remove('O')
        
    label_list = sorted(list(all_labels))
    
    # print(f"\nEvaluating on {len(dataset)} samples with {len(label_list)} labels: {label_list}")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating", leave=False):
        batch_items = dataset[i:i+batch_size]
        batch_texts = [" ".join(d['tokenized_text']) for d in batch_items]
        
        with torch.no_grad():
            if hasattr(model, 'inference'):
                 batch_preds = model.inference(batch_texts, label_list, threshold=0.5)
            elif hasattr(model, 'batch_predict_entities'):
                 batch_preds = model.batch_predict_entities(batch_texts, label_list, threshold=0.5)
            else:
                 batch_preds = [model.predict_entities(t, label_list, threshold=0.5) for t in batch_texts]

        for idx, item in enumerate(batch_items):
            # Ground Truth Spans
            gt_spans = set()
            for s, e, l in item['ner']:
                if l != 'O': # Ignore O class
                    gt_spans.add((l, s, e)) 
                    support[l] += 1
            
            preds = batch_preds[idx]
            
            # Map character spans to token spans
            tokens = item['tokenized_text']
            char_to_token = {}
            cursor = 0
            for t_i, token in enumerate(tokens):
                start = cursor
                end = cursor + len(token)
                for c in range(start, end):
                    char_to_token[c] = t_i
                cursor = end + 1 # +1 for space
            
            pred_spans = set()
            for p in preds:
                label = p['label']
                if label == 'O': continue # Explicitly ignore predicted O just in case

                p_start = p['start']
                p_end = p['end'] 
                
                if p_start in char_to_token and (p_end - 1) in char_to_token:
                    t_start = char_to_token[p_start]
                    t_end = char_to_token[p_end - 1]
                    pred_spans.add((label, t_start, t_end))
            
            # Compare
            tps = pred_spans.intersection(gt_spans)
            fps = pred_spans - gt_spans
            fns = gt_spans - pred_spans
            
            for l, s, e in tps: tp[l] += 1
            for l, s, e in fps: fp[l] += 1
            for l, s, e in fns: fn[l] += 1

    # Calculate Global Metrics
    p_s, r_s, f1_s = [], [], []
    valid_labels = [l for l in label_list if support[l] > 0 or fp[l] > 0] 
    
    for l in valid_labels:
        t = tp[l]
        f_p = fp[l]
        f_n = fn[l]
        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_s.append(p)
        r_s.append(r)
        f1_s.append(f1)
        
    macro_p = np.mean(p_s) if p_s else 0.0
    macro_r = np.mean(r_s) if r_s else 0.0
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values()) 
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print(f"\n[Validation] Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "micro_p": micro_p,
        "micro_r": micro_r
    }

# ==========================================
# DATA LOADING & CONVERSION
# ==========================================

print("Loading datasets and mappings...")
print(f"\nDataset folder: {DATASET_FOLDER if not is_running_on_kaggle() else 'Kaggle'}")
print(f"Train path: {train_path}")
print(f"Validation path: {val_path}")
print(f"Test path: {test_path}")

with open(train_path, "r") as f:
    train_dataset = json.load(f)

with open(val_path, "r") as f:
    val_dataset = json.load(f)

with open(test_path, "r") as f:
    test_dataset = json.load(f)

with open(label2id_path, "r") as f:
    label2id = json.load(f)

id2label = {str(v): k for k, v in label2id.items()}

def convert_ids_to_labels(dataset, id_map):
    new_dataset = []
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            label_id = str(label_id)
            if label_id in id_map:
                label_name = id_map[label_id]
                if label_name == "O": 
                    continue
                new_ner.append([start, end, label_name])
            else:
                 print(f"Warning: Label ID not found in map: {label_id}")
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset

print("\nConverting Training Dataset IDs to Labels...")
train_dataset = convert_ids_to_labels(train_dataset, id2label)

print("\nConverting Validation Dataset IDs to Labels...")
val_dataset = convert_ids_to_labels(val_dataset, id2label)

print("\nConverting Test Dataset IDs to Labels...")
test_dataset = convert_ids_to_labels(test_dataset, id2label)

print(f'\nFinal Train dataset size: {len(train_dataset)}')
print(f'Final Validation dataset size: {len(val_dataset)}')
print(f'Final Test dataset size: {len(test_dataset)}')

# ==========================================
# OPTUNA OPTIMIZATION
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/glinerbismall2/'
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

data_size = len(train_dataset)
num_batches_per_epoch = data_size // batch_size
max_steps = -1 
save_steps = 100
logging_steps = save_steps

def objective(trial):
    # Propose Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "constant_with_warmup"])
    focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 2, 5.0)  # New: optimize gamma
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"   LR: {learning_rate}")
    print(f"   Scheduler: {lr_scheduler_type}")
    print(f"   Focal Loss Gamma: {focal_loss_gamma}")

    # Load fresh model
    model = GLiNER.from_pretrained(MODEL_NAME).to(device)

    # Unique output directory for this trial
    trial_output_dir = f"optuna_models/trial_{trial.number}"
    
    # Train
    try:
        trainer = model.train_model(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=trial_output_dir,
            learning_rate=learning_rate,       # <--- Optimized
            lr_scheduler_type=lr_scheduler_type, # <--- Optimized
            weight_decay=0.01,
            others_lr=1e-5,
            others_weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            focal_loss_alpha=0.75,
            focal_loss_gamma=focal_loss_gamma,   # <--- Optimized
            num_train_epochs=target_epochs,
            max_steps=max_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=2,
            eval_strategy="steps",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            use_cpu=not use_cuda,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        # Prune this trial if it crashes (e.g. OOM or NaN)
        raise optuna.exceptions.TrialPruned()

    # Evaluation on the loaded best model (using validation dataset)
    # Note: 'model' object is updated in-place by train_model wrapper usually, 
    # but load_best_model_at_end=True ensures it holds best weights at end.
    
    # Estrai la miglior validation loss dal trainer (salvata automaticamente grazie a load_best_model_at_end=True)
    val_loss = trainer.state.best_metric if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_metric') else None
    
    if val_loss is None:
        # Fallback se best_metric non è disponibile (es. versioni vecchie di transformers)
        print("⚠️ Warning: Could not extract best_metric from trainer state. Checking log history...")
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
             for log in reversed(trainer.state.log_history):
                 if 'eval_loss' in log:
                     val_loss = log['eval_loss']
                     break
    
    if val_loss is None:
        val_loss = float('inf') # Should not happen if training succeeded
        
    print(f"   Best Validation Loss: {val_loss:.4f}")
    
    # Clean up artifacts to save space (Except maybe the best one? User might want to inspect)
    # For now we delete the checkpoints to avoid filling disk
    try:
        shutil.rmtree(trial_output_dir)
    except:
        pass
    
    return val_loss

# Run Optimization with Pruning
# MedianPruner stops unpromising trials after epoch 2 (n_startup_trials=5, n_warmup_steps=2)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,  # No pruning for first 5 trials (gathering baseline)
    n_warmup_steps=2,    # Prune after 2 epochs (step 2)
    interval_steps=1     # Check every epoch
)

study = optuna.create_study(
    direction="minimize", 
    study_name="gliner_lr_scheduler_gamma_opt",
    pruner=pruner
)
study.optimize(objective, n_trials=20)

# ==========================================
# SAVE RESULTS WITH TIMESTAMP
# ==========================================
import time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory
output_dir = "finetune/optuna_results"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*50)
print("OPTIMIZATION FINISHED")
print("="*50)
print(f"Best Trial: {study.best_trial.number}")
print(f"Best Value (Validation Loss): {study.best_value}")
print(f"Best Params: {study.best_params}")

# ==========================================
# 1. GENERATE VISUALIZATION (PNG)
# ==========================================
print(f"\n📊 Generating visualization plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Optimization History
trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_numbers = [t.number for t in trials]
trial_values = [t.value for t in trials]

ax1.scatter(trial_numbers, trial_values, alpha=0.6, s=50)
ax1.set_xlabel('Trial', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Optimization History', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add best-so-far line (minimization)
best_so_far = []
current_best = float('inf')
for val in trial_values:
    if val < current_best:
        current_best = val
    best_so_far.append(current_best)
ax1.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='Best so far')
ax1.legend()

# Right plot: Hyperparameter Importance
try:
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())
    
    ax2.barh(params, values)
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
except Exception as e:
    ax2.text(0.5, 0.5, f'Importance calculation\nrequires completed trials\n({str(e)})', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')

plt.tight_layout()

# Save PNG
png_filename = f"optuna_plots_{timestamp}.png"
png_path = os.path.join(output_dir, png_filename)
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Plots saved to: {png_path}")

# ==========================================
# 2. SAVE COMPREHENSIVE JSON
# ==========================================
print(f"\n💾 Saving comprehensive study results...")

# Collect all trials data
all_trials_data = []
completed_trials = 0
pruned_trials = 0

for trial in study.trials:
    trial_data = {
        "number": trial.number,
        "value": trial.value,
        "state": str(trial.state),
        "params": trial.params
    }
    all_trials_data.append(trial_data)
    
    if trial.state == optuna.trial.TrialState.COMPLETE:
        completed_trials += 1
    elif trial.state == optuna.trial.TrialState.PRUNED:
        pruned_trials += 1

# Calculate total time (if available)
total_time = 0
if len(study.trials) > 0:
    first_trial = min(study.trials, key=lambda t: t.datetime_start)
    last_trial = max(study.trials, key=lambda t: t.datetime_complete if t.datetime_complete else t.datetime_start)
    if first_trial.datetime_start and last_trial.datetime_complete:
        total_time = (last_trial.datetime_complete - first_trial.datetime_start).total_seconds()

# Create comprehensive JSON structure
study_data = {
    "study_name": study.study_name,
    "timestamp": timestamp,
    "n_trials": len(study.trials),
    "completed_trials": completed_trials,
    "pruned_trials": pruned_trials,
    "total_time_seconds": total_time,
    "best_trial": {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_params
    },
    "all_trials": all_trials_data,
    "fixed_params": {
        "epochs": target_epochs,
        "batch_size": batch_size,
        "focal_loss_alpha": 0.75,
        "weight_decay": 0.01,
        "others_lr": 1e-5,
        "warmup_ratio": 0.1,
        "model_name": MODEL_NAME,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "dataset_folder": DATASET_FOLDER if not is_running_on_kaggle() else "Kaggle"
    }
}

# Save comprehensive JSON
json_filename = f"optuna_study_{timestamp}.json"
json_path = os.path.join(output_dir, json_filename)
with open(json_path, "w") as f:
    json.dump(study_data, f, indent=2)

print(f"✅ Study data saved to: {json_path}")

# ==========================================
# 3. SAVE BEST PARAMS (separate file for convenience)
# ==========================================
best_params_filename = f"best_params_{timestamp}.json"
best_params_path = os.path.join(output_dir, best_params_filename)
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=4)

print(f"✅ Best parameters saved to: {best_params_path}")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*50)
print("📁 OUTPUT SUMMARY")
print("="*50)
print(f"Directory: {output_dir}/")
print(f"  • {png_filename}")
print(f"  • {json_filename}")
print(f"  • {best_params_filename}")
print("\n🎯 OPTIMIZATION PARAMETERS:")
print(f"  • learning_rate: 1e-6 → 1e-4")
print(f"  • lr_scheduler_type: [linear, cosine, constant_with_warmup]")
print(f"  • focal_loss_gamma: 2.0 → 5.0")
print(f"\n📊 BEST RESULT:")
print(f"  Trial #{study.best_trial.number}")
print(f"  Validation Loss: {study.best_value:.4f}")
print(f"  Params: {study.best_params}")
print("="*50)

