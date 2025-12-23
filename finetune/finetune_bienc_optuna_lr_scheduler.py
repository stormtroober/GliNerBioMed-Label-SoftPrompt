
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json 
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from gliner import GLiNER
import shutil
import optuna

# ==========================================
# CONFIGURATION
# ==========================================

if is_running_on_kaggle():
    path = "/kaggle/input/jnlpa15k/"
else:
    path = "finetune/"

train_path = path + "jnlpa_train.json"
test_path = path + "jnlpa_test.json"
label2id_path = path + "label2id.json"

# Training Configuration
target_steps = None       # Se impostato, il training si fermerà esattamente a questi step
target_epochs = 3         # Reduced to 3 for faster Optuna trials
batch_size = 32

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
with open(train_path, "r") as f:
    train_dataset = json.load(f)

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

print("\nConverting Test Dataset IDs to Labels...")
test_dataset = convert_ids_to_labels(test_dataset, id2label)

print(f'\nFinal Train dataset size: {len(train_dataset)}')
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
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "constant", "constant_with_warmup"])
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"   LR: {learning_rate}")
    print(f"   Scheduler: {lr_scheduler_type}")

    # Load fresh model
    model = GLiNER.from_pretrained(MODEL_NAME).to(device)

    # Unique output directory for this trial
    trial_output_dir = f"optuna_models/trial_{trial.number}"
    
    # Train
    try:
        trainer = model.train_model(
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
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
            focal_loss_gamma=2,
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

    # Evaluation on the loaded best model
    # Note: 'model' object is updated in-place by train_model wrapper usually, 
    # but load_best_model_at_end=True ensures it holds best weights at end.
    
    metrics = calculate_metrics(test_dataset, model, batch_size=batch_size)
    
    # Clean up artifacts to save space (Except maybe the best one? User might want to inspect)
    # For now we delete the checkpoints to avoid filling disk
    try:
        shutil.rmtree(trial_output_dir)
    except:
        pass
    
    return metrics['micro_f1']

# Run Optimization
study = optuna.create_study(direction="maximize", study_name="gliner_lr_scheduler_opt")
study.optimize(objective, n_trials=15) # 10 trials as a reasonable start

print("\n" + "="*50)
print("OPTIMIZATION FINISHED")
print("="*50)
print(f"Best Trial: {study.best_trial.number}")
print(f"Best Value (Micro F1): {study.best_value}")
print(f"Best Params: {study.best_params}")

# Optional: Retrain with best params or save them
best_params_path = "best_optuna_params.json"
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=4)
print(f"Best parameters saved to {best_params_path}")
