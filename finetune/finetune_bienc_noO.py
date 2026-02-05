
#Time per epoch: 133.13s (2.22 min)

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json 
import random
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from gliner import GLiNER
import shutil

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ==========================================
# CONFIGURATION
# ==========================================

# Dataset Configuration (choose one)
# For JNLPA dataset
DATASET_FOLDER = "dataset"  # or "dataset_bc5cdr" for BC5CDR dataset

if is_running_on_kaggle():
    path = "/kaggle/input/jnlpa-18-5k15-3-5-complete/"
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
target_epochs = 10         # Usato solo se target_steps è None
batch_size = 32

# ==========================================
# METRICS FUNCTION
# ==========================================
def calculate_metrics(dataset, model, batch_size=1):
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
    
    print(f"\nEvaluating on {len(dataset)} samples with {len(label_list)} labels: {label_list}")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
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
    total_fn = sum(fn.values()) # Should this be based on support? Typically recall denonimator is Total Positives = TP + FN
    # Actually micro_r denominator is Total True Positives available = TP + FN
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print("\n## 📈 Global Metrics (Label2ID Mode, EXCLUDING O)\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

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
    converted_count = 0
    filtered_count = 0
    total_spans = 0
    
    new_dataset = []
    
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            total_spans += 1
            
            # label_id should be a string representing the ID (e.g. "0")
            label_id = str(label_id)
            
            if label_id in id_map:
                label_name = id_map[label_id]
                
                # Check for "O" class
                if label_name == "O": 
                    filtered_count += 1
                    continue
                    
                new_ner.append([start, end, label_name])
                converted_count += 1
            else:
                 print(f"Warning: Label ID not found in map: {label_id}")
    
        item['ner'] = new_ner
        new_dataset.append(item)
        
    print(f"  Processed {len(dataset)} items.")
    print(f"  Total Spans: {total_spans}")
    print(f"  Converted (ID -> Label): {converted_count}")
    print(f"  Filtered (O class): {filtered_count}")
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
# MODEL SETUP
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/glinerbismall2/'
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GLiNER.from_pretrained(MODEL_NAME).to(device)
print(f"Model loaded on {device}")

print("\n" + "="*50)
print("BASELINE EVALUATION (Pre-Training / Zero-shot with Short Labels)")
print("="*50)
calculate_metrics(test_dataset, model)

# ==========================================
# TRAINING
# ==========================================
data_size = len(train_dataset)
num_batches_per_epoch = data_size // batch_size

print("\n" + "="*50)
print("TRAINING CONFIGURATION")
print("="*50)
print(f'Mode: EPOCHS (Target: {target_epochs})')
total_steps = target_epochs * num_batches_per_epoch
print(f'Total Steps: {total_steps}')

max_steps = -1 
num_train_epochs = target_epochs

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

save_steps = 100
logging_steps = save_steps

print(f"\n⏱️ TRAINING TIMING INFO:")
print(f"  Samples: {data_size}")
print(f"  Batch size: {batch_size}")
print(f"  Steps per epoch: {num_batches_per_epoch}")
print(f"  Logging every: {logging_steps} steps")
print(f"  Saving every: {save_steps} steps")
print(f"  Total epochs: {num_train_epochs}")
print(f"  Total steps: {total_steps}\n")

# Start timing
start_time = time.time()

trainer = model.train_model(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # ✅ Usa validation set, NON test set
    output_dir="models_short_label",
    learning_rate=9.96554625802328e-05,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=4.414064315327594,
    num_train_epochs=num_train_epochs,
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

# Calculate and display timing statistics
total_time = time.time() - start_time
time_per_epoch = total_time / num_train_epochs
time_per_step = total_time / total_steps

print(f"\n⏱️ TRAINING COMPLETE:")
print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"  Time per epoch: {time_per_epoch:.2f}s ({time_per_epoch/60:.2f} min)")
print(f"  Time per step: {time_per_step:.2f}s\n")

# Save the best model
best_model_path = "models_short_label/best_model"
model.save_pretrained(best_model_path)
print(f"Best model saved to {best_model_path}")

# Create timestamp for checkpoint
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a comprehensive checkpoint dictionary (same format as monoenc)
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": model.config,
    "training_metadata": {
        "base_model_name": MODEL_NAME,
        "encoder_type": "bi-encoder",
        "dataset_name": DATASET_FOLDER if not is_running_on_kaggle() else "jnlpa-18-5k15-3-5-complete",
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset),
        "num_epochs": num_train_epochs,
        "batch_size": batch_size,
        "learning_rate": 9.96554625802328e-05,
        "weight_decay": 0.01,
        "others_lr": 1e-5,
        "others_weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "focal_loss_alpha": 0.75,
        "focal_loss_gamma": 4.414064315327594,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "random_seed": RANDOM_SEED,
        "total_time_seconds": total_time,
        "time_per_epoch_seconds": time_per_epoch,
        "time_per_step_seconds": time_per_step,
        "total_steps": total_steps,
        "exclude_O_class": True,
        "timestamp": timestamp
    }
}

# Save the complete checkpoint as .pt file
checkpoint_filename = f"finetune_bienc_{timestamp}.pt"
checkpoint_path = os.path.join("models_short_label", checkpoint_filename)

if not os.path.exists("models_short_label"):
    os.makedirs("models_short_label")

torch.save(checkpoint, checkpoint_path)
print(f"Best model checkpoint (.pt) saved to {checkpoint_path}")

# Also save metadata as JSON for easy reading
json_filename = f"metadata_{timestamp}.json"
json_path = os.path.join("models_short_label", json_filename)
metadata_json = checkpoint["training_metadata"].copy()
metadata_json["label2id"] = label2id
with open(json_path, "w") as f:
    json.dump(metadata_json, f, indent=2)
print(f"Metadata JSON saved to {json_path}")

# Load and Evaluate
trained_model = GLiNER.from_pretrained(best_model_path).to(device)
print("\n" + "="*50)
print("FINAL EVALUATION (Post-Training)")
print("="*50)
calculate_metrics(test_dataset, trained_model)
