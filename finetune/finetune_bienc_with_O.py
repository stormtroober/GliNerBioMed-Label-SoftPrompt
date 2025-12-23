
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

# ==========================================
# CONFIGURATION
# ==========================================
train_path = "finetune/jnlpa_train.json"
test_path = "finetune/jnlpa_test.json"
label2desc_path = "label2desc.json"
label2id_path = "label2id.json"

# Training Configuration
target_steps = None       
target_epochs = 3         
batch_size = 8

# ==========================================
# METRICS FUNCTION (Modified: Train on O, Evaluate NO O)
# ==========================================
def calculate_metrics(dataset, model, batch_size=1):
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Gather all labels from dataset 
    all_labels = set()
    for d in dataset:
        for x in d['ner']: 
            all_labels.add(x[2])
    
    # We pass ALL labels (including O) to the model for inference
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
                # We load ALL spans, but we only care about non-O for metrics
                if l != "O":
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
                # IGNORING PREDICTED 'O'
                if label == "O":
                    continue

                p_start = p['start']
                p_end = p['end'] 
                
                if p_start in char_to_token and (p_end - 1) in char_to_token:
                    t_start = char_to_token[p_start]
                    t_end = char_to_token[p_end - 1]
                    pred_spans.add((label, t_start, t_end))
            
            # Compare (Only non-O entities remain here)
            tps = pred_spans.intersection(gt_spans)
            fps = pred_spans - gt_spans
            fns = gt_spans - pred_spans
            
            for l, s, e in tps: tp[l] += 1
            for l, s, e in fps: fp[l] += 1
            for l, s, e in fns: fn[l] += 1

    # Calculate Global Metrics
    p_s, r_s, f1_s = [], [], []
    
    # Filter 'O' from the labels to report
    valid_labels = [l for l in label_list if l != "O" and (support[l] > 0 or fp[l] > 0)]
    
    for l in valid_labels:
        t = tp[l]
        f_p = fp[l]
        f_n = fn[l]
        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        print(f"  > Class '{l}': P={p:.2f}, R={r:.2f}, F1={f1:.2f} (Supp: {support[l]})")
        
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
    
    print("\n## 📈 Global Metrics (EXCLUDING 'O')\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

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
    converted_count = 0
    total_spans = 0
    kept_o_count = 0
    
    new_dataset = []
    
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            total_spans += 1
            label_id = str(label_id)
            
            if label_id in id_map:
                label_name = id_map[label_id]
                
                # NOTE: NOT filtering O anymore!
                if label_name == "O":
                    kept_o_count += 1
                    
                new_ner.append([start, end, label_name])
                converted_count += 1
            else:
                 print(f"Warning: Label ID not found in map: {label_id}")
    
        item['ner'] = new_ner
        new_dataset.append(item)
        
    print(f"  Processed {len(dataset)} items.")
    print(f"  Total Spans: {total_spans}")
    print(f"  Converted: {converted_count}")
    print(f"  'O' Class Kept: {kept_o_count}")
    return new_dataset

print("\nConverting Training Dataset IDs to Labels (WITH O)...")
train_dataset = convert_ids_to_labels(train_dataset, id2label)

print("\nConverting Test Dataset IDs to Labels (WITH O)...")
test_dataset = convert_ids_to_labels(test_dataset, id2label)

print(f'\nFinal Train dataset size: {len(train_dataset)}')
print(f'Final Test dataset size: {len(test_dataset)}')

# ==========================================
# MODEL SETUP
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/gliner2-1small/'
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GLiNER.from_pretrained(MODEL_NAME).to(device)
print(f"Model loaded on {device}")

print("\n" + "="*50)
print("BASELINE EVALUATION WITH 'O' (Filtered Metrics)")
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
save_steps = min(50, total_steps) if total_steps > 0 else 100

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

trainer = model.train_model(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    output_dir="models_short_label_with_O_filtered",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    save_steps=save_steps,
    save_total_limit=2,
    eval_strategy="steps",
    load_best_model_at_end=True,
    dataloader_num_workers=0,
    use_cpu=not use_cuda,
    bf16=use_bf16,
    fp16=use_fp16,
    report_to="none",
    )

# Save the best model
best_model_path = "models_short_label_with_O_filtered/best_model"
model.save_pretrained(best_model_path)
print(f"Best model saved to {best_model_path}")

# Zip it
zip_name = "best_model_short_label_with_O_filtered_archive"
shutil.make_archive(zip_name, 'zip', best_model_path)
print(f"Best model zipped to {os.path.abspath(zip_name + '.zip')}")

# Load and Evaluate
trained_model = GLiNER.from_pretrained(best_model_path).to(device)
print("\n" + "="*50)
print("FINAL EVALUATION WITH 'O' (Post-Training, Filtered Metrics)")
print("="*50)
calculate_metrics(test_dataset, trained_model)
