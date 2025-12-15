import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

import json 
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def calculate_metrics(dataset, model, batch_size=1):
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Gather all labels from dataset
    all_labels = set()
    for d in dataset:
        for x in d['ner']: all_labels.add(x[2])
    label_list = sorted(list(all_labels))
    
    print(f"\nEvaluating on {len(dataset)} samples with {len(label_list)} labels...")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_items = dataset[i:i+batch_size]
        batch_texts = [" ".join(d['tokenized_text']) for d in batch_items]
        
        # Use batch_predict_entities if available for speed (encodes labels once per batch)
        # Wrap in no_grad to save memory
        with torch.no_grad():
            if hasattr(model, 'inference'):
                 batch_preds = model.inference(batch_texts, label_list, threshold=0.5)
            elif hasattr(model, 'batch_predict_entities'):
                 batch_preds = model.batch_predict_entities(batch_texts, label_list, threshold=0.5)
            else:
                 # Fallback to single if batch not found
                 batch_preds = [model.predict_entities(t, label_list, threshold=0.5) for t in batch_texts]

        for idx, item in enumerate(batch_items):
            tokens = item['tokenized_text']
            
            # Ground Truth Spans
            gt_spans = set()
            for s, e, l in item['ner']:
                gt_spans.add((l, s, e)) 
                support[l] += 1
            
            preds = batch_preds[idx]
            
            # Map character spans to token spans
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
        
    # Macro
    macro_p = np.mean(p_s) if p_s else 0.0
    macro_r = np.mean(r_s) if r_s else 0.0
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    
    # Micro
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    # Weighted
    total_support = sum(support.values())
    if total_support > 0:
        weighted_p = sum(p * support[l] for p, l in zip(p_s, valid_labels)) / total_support
        weighted_r = sum(r * support[l] for r, l in zip(r_s, valid_labels)) / total_support
        weighted_f1 = sum(f * support[l] for f, l in zip(f1_s, valid_labels)) / total_support
    else:
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    print("\n## 📈 Metriche Globali (ESCLUSO 'O')\n")
    print(f"### Riassunto Performance")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")
    print(f"| Weighted     | {weighted_p:.4f} | {weighted_r:.4f} | {weighted_f1:.4f} |")

train_path = "/kaggle/input/example/data.json"

with open(train_path, "r") as f:
    data = json.load(f)

print('Dataset size:', len(data))

random.shuffle(data)
data = data[:1000]
print('Dataset reduced to:', len(data))
print('Dataset is shuffled...')

train_dataset = data[:int(len(data)*0.9)]
test_dataset = data[int(len(data)*0.9):]

print('Dataset is splitted...')

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from gliner import GLiNER


if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/gliner2-1small/'
else:
    MODEL_NAME = "urchade/gliner_small-v2.1"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GLiNER.from_pretrained(MODEL_NAME).to(device)
print(f"Model loaded on {device}")

print("\n" + "="*50)
print("BASELINE EVALUATION (Pre-Training)")
print("="*50)
calculate_metrics(test_dataset, model)

# Training Configuration
target_steps = None       # Se impostato, il training si fermerà esattamente a questi step
target_epochs = 3    # Usato solo se target_steps è None

batch_size = 16
data_size = len(train_dataset)
num_batches_per_epoch = data_size // batch_size

print("\n" + "="*50)
print("TRAINING CONFIGURATION")
print("="*50)
print(f'Train dataset size: {data_size}')
print(f'Batch size: {batch_size}')
print(f'Steps per epoch: {num_batches_per_epoch}')

if target_steps:
    max_steps = target_steps
    num_train_epochs = target_steps / num_batches_per_epoch
    print(f'Mode: STEPS (Target: {target_steps})')
    print(f'Equivalent Epochs: {num_train_epochs:.4f}')
else:
    # Default to 1 epoch if neither is set properly, or use target_epochs
    train_epochs = target_epochs if target_epochs else 1
    max_steps = -1 # Trainer will use epochs
    num_train_epochs = train_epochs
    total_steps = train_epochs * num_batches_per_epoch
    print(f'Mode: EPOCHS (Target: {train_epochs})')
    print(f'Total Steps: {total_steps}')

save_steps = min(50, max_steps) if max_steps > 0 else 100
print(f'Save checkpoint every: {save_steps} steps')
print("="*50 + "\n")

# Check for GPU and BF16 support
use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

import shutil

trainer = model.train_model(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear", #cosine
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

# Save the best model explicitly
best_model_path = "models/best_model"
model.save_pretrained(best_model_path)
print(f"Best model saved to {best_model_path}")

# Zip only the best model directory
zip_name = "best_model_archive"
shutil.make_archive(zip_name, 'zip', best_model_path)
print(f"Best model zipped to {os.path.abspath(zip_name + '.zip')}")

# After training, load the best model
trained_model = GLiNER.from_pretrained(best_model_path).to(device)
print(f"Best model loaded from {best_model_path}")

# ==========================================
# EVALUATION ON TEST DATASET
# ==========================================

print("\n" + "="*50)
print("FINAL EVALUATION (Post-Training)")
print("="*50)
calculate_metrics(test_dataset, trained_model)
