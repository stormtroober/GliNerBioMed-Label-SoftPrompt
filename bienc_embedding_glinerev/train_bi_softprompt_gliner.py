
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
import torch.nn as nn
from gliner import GLiNER
import shutil

# ==========================================
# CONFIGURATION
# ==========================================

if is_running_on_kaggle():
    path = "/kaggle/input/spanbi16k/"
else:
    path = "../dataset/"

train_path = path + "dataset_span_bi.json"
test_path = path + "test_dataset_span_bi.json"
label2id_path = path + "label2id.json"

# Training Configuration
target_steps = None       # Se impostato, il training si fermerà esattamente a questi step
target_epochs = 5         # Usato solo se target_steps è None
batch_size = 8

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
with open(train_path, "r") as f:
    train_dataset = json.load(f)

with open(test_path, "r") as f:
    test_dataset = json.load(f)

with open(label2id_path, "r") as f:
    label2id = json.load(f)

# Create reverse mapping: ID -> Label Name
# Ensure keys in map are strings because JSON keys are strings, 
# but values in label2id are ints. 
# The dataset has IDs as strings (from my previous edit).
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

print("\nConverting Test Dataset IDs to Labels...")
test_dataset = convert_ids_to_labels(test_dataset, id2label)

print(f'\nFinal Train dataset size: {len(train_dataset)}')
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
# SOFT PROMPT INJECTION (Monkey Patch)
# ==========================================
print("\n" + "="*50)
print("INJECTING SOFT PROMPT ENCODER")
print("="*50)

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, hidden_dim=None, dropout=0.1, tokenizer=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        if hidden_dim is None: hidden_dim = embed_dim # REDUCED from *4 to *1 for regularization
            
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._debug_printed = False
        self.tokenizer = tokenizer

    def forward(self, input_ids):
        if not self._debug_printed:
            print("\n[DEBUG] MLPPromptEncoder First Forward Pass")
            print("==========================================")
            ids = input_ids[0].tolist()
            token_strings = []
            if self.tokenizer:
                token_strings = self.tokenizer.convert_ids_to_tokens(ids)
            
            print(f"Input IDs (First Sequence): {ids}")
            
        # 1. Get original embeddings
        x = self.embedding(input_ids)
        
        # 2. Compute Transform
        delta = self.mlp(x)
        
        # 3. Apply Mask: Do NOT transform [CLS](101), [SEP](102), [PAD](0)
        mask = (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
        mask_float = mask.unsqueeze(-1).expand_as(delta).float()
        
        if not self._debug_printed:
            m_vals = mask[0].tolist()
            print(f"Mask Values (0=Original, 1=SoftPrompt): {m_vals}")
            print("Interpretation:")
            for i, (tid, m) in enumerate(zip(ids, m_vals)):
                token_label = "???"
                if token_strings:
                     token_label = token_strings[i]
                else:
                    if tid == 101: token_label = "[CLS]"
                    elif tid == 102: token_label = "[SEP]"
                    elif tid == 0: token_label = "[PAD]"
                    else: token_label = "WORD"
                
                action = "IDENTITY (Original)" if m == 0 else "SOFT PROMPT (Modified)"
                print(f"  Pos {i}: ID {tid:<5} ({token_label:<10}) -> Mask {m} -> {action}")
            print("==========================================\n")
            self._debug_printed = True
        
        # Apply mask
        delta = delta * mask_float
        
        return self.norm(x + delta)

class SoftPromptLabelEncoderWrapper(nn.Module):
    def __init__(self, original_encoder, prompt_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        # Intercept inputs
        if inputs_embeds is None and input_ids is not None:
            # Generate soft embeddings from input_ids using our Prompt Encoder
            inputs_embeds = self.prompt_encoder(input_ids)
        
        # Delegate to original encoder
        return self.original_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute access to original encoder if not found here
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_encoder, name)

# 1. Setup Wrapper
lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
original_embeddings = lbl_enc_model.embeddings.word_embeddings
vocab_size = original_embeddings.num_embeddings
embed_dim = original_embeddings.embedding_dim

# Load Tokenizer for Debugging
from transformers import AutoTokenizer
try:
    # Use the same tokenizer as the model backbone (usually BERT-like for all-MiniLM)
    debug_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Could not load debug tokenizer: {e}")
    debug_tokenizer = None

prompt_encoder = MLPPromptEncoder(
    original_embeddings, 
    vocab_size, 
    embed_dim, 
    dropout=0.1,
    tokenizer=debug_tokenizer
).to(device)

wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)

# 2. Inject Wrapper
model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
print("✅ SoftPromptLabelEncoderWrapper injected into model.")

# 3. Configure Freezing
# ==========================================
# 3. UNFREEZE EVERYTHING (Full Fine-Tuning)
# ==========================================
# User requested to train EVERYTHING, similar to standard fine-tuning, 
# but with our Soft Prompt injection active.

for p in model.parameters():
    p.requires_grad = True

print("===========")
print("INJECTING SOFT PROMPT ENCODER")
print("===========")
print("✅ SoftPromptLabelEncoderWrapper injected into model.")
print("🔓 UNFROZEN EVERYTHING: Text Encoder, Label Encoder, Prompt Encoder, Projection.")
print("   (Standard Fine-Tuning + Soft Prompts)")

# Verify
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
print("🔓 Prompt Encoder and Projection Layer Unfrozen.")

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

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

save_steps = 50
logging_steps = save_steps

trainer = model.train_model(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    output_dir="models_short_label",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=5e-6, # REDUCED from 1e-5
    others_weight_decay=0.05, # INCREASED from 0.01
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
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

# Save FULL model state dict to 'savings' folder with TIMESTAMP
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
savings_dir = "savings"
if not os.path.exists(savings_dir):
    os.makedirs(savings_dir)

best_model_name = f"best_softprompt_model_{timestamp}.pt"
best_model_path = os.path.join(savings_dir, best_model_name)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': embed_dim, # Updated
        'dropout': 0.1, # Updated
        'num_train_epochs': num_train_epochs,
        'learning_rate': 5e-6,
        'weight_decay': 0.01,
        'per_device_train_batch_size': batch_size,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 2
    }
}, best_model_path)

print(f"✅ Best Full Model (Soft Prompt) saved to {best_model_path}")

# Load and Evaluate
# NOTE: We skip reloading from disk for immediate eval.
# The 'model' variable currently holds the best state because load_best_model_at_end=True
trained_model = model 
print("\n" + "="*50)
print("FINAL EVALUATION (Post-Training)")
print("="*50)
calculate_metrics(test_dataset, trained_model)
