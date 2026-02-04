import os
import json
import random
import logging
import argparse
import time
import datetime
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Union, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Imports GLiNER e Transformers
from gliner import GLiNER
from gliner.training import Trainer as BaseTrainer
from gliner.training import TrainingArguments as BaseTrainingArguments

# --- GESTIONE IMPORT TRANSFORMERS ---
try:
    from transformers.pytorch_utils import get_parameter_names
except ImportError:
    try:
        from transformers.trainer import get_parameter_names
    except ImportError:
        def get_parameter_names(model, forbidden_layer_types):
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            result += list(model._parameters.keys())
            return result

try:
    from transformers.utils import is_sagemaker_mp_enabled
except ImportError:
    try:
        from transformers.trainer import is_sagemaker_mp_enabled
    except ImportError:
        def is_sagemaker_mp_enabled():
            return False
# ------------------------------------

# ==========================================
# 0. CONFIGURAZIONE AMBIENTE
# ==========================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 1. ARCHITETTURA (PROMPT 384 -> PROJECTION 768)
# ==========================================

class PromptEncoder(nn.Module):
    def __init__(self, backbone_hidden_size: int, num_labels: int, soft_prompt_length: int = 1, mid_dim: int = None):
        super().__init__()
        self.backbone_hidden_size = backbone_hidden_size
        self.num_labels = num_labels
        self.mlp = nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size)
        self.norm = nn.LayerNorm(self.backbone_hidden_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch_embeddings: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        modified_embeddings = batch_embeddings.clone()
        num_labels = batch_embeddings.shape[0]

        for label_idx in range(num_labels):
            label_positions = (position_mask[label_idx] == 1).nonzero(as_tuple=True)[0]
            if len(label_positions) == 0: continue
            label_embeddings = batch_embeddings[label_idx, label_positions, :] 
            soft_prompt_embeddings = self.mlp(label_embeddings)
            modified_embeddings[label_idx, label_positions, :] = self.norm(
                label_embeddings + soft_prompt_embeddings
            )
        return modified_embeddings

# ==========================================
# 1.5. LABEL ENCODER WRAPPER (approccio v1 che funziona)
# ==========================================

class SoftPromptLabelEncoderWrapper(nn.Module):
    """Wrapper che intercetta le chiamate al label encoder e inietta soft prompts.
    
    Questo approccio mantiene il modello GLiNER intatto e modifica solo
    il punto di iniezione delle embeddings.
    """
    def __init__(self, original_encoder, prompt_encoder, tokenizer=None):
        super().__init__()
        self.original_encoder = original_encoder
        self.prompt_encoder = prompt_encoder
        self.tokenizer = tokenizer
        self._debug_printed = False
        
    def _create_position_mask(self, input_ids):
        """Crea mask per identificare i token delle label (escludendo special tokens)."""
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 101, 102, 0
        mask = (input_ids != BOS_TOKEN) & (input_ids != EOS_TOKEN) & (input_ids != PAD_TOKEN)
        return mask.long()
        
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        # Se già abbiamo embeddings, usa quelle
        if inputs_embeds is None and input_ids is not None:
            # 1. Ottieni embeddings originali
            raw_embeds = self.original_encoder.embeddings(input_ids)
            
            # 2. Crea position mask
            position_mask = self._create_position_mask(input_ids)
            
            # 3. Applica soft prompt
            inputs_embeds = self.prompt_encoder(raw_embeds, position_mask)
            
            # Debug
            if not self._debug_printed:
                print(f"\n[WRAPPER DEBUG]")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Raw embeds shape: {raw_embeds.shape}")
                print(f"  Soft embeds shape: {inputs_embeds.shape}")
                self._debug_printed = True
        
        # 4. Passa al transformer encoder originale
        return self.original_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def __getattr__(self, name):
        """Delega tutti gli altri attributi all'encoder originale."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_encoder, name)


# ==========================================
# 2. TRAINER & UTILS
# ==========================================

@dataclass
class SoftGlinerTrainingArguments(BaseTrainingArguments):
    prompt_encoder_lr: Optional[float] = None

class SoftGlinerOptimizer(BaseTrainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            prompt_params = [n for n, _ in opt_model.named_parameters() if "prompt_encoder" in n]
            
            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters() if n in prompt_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay,
                 "lr": self.args.prompt_encoder_lr if self.args.prompt_encoder_lr else self.args.learning_rate},
                {"params": [p for n, p in opt_model.named_parameters() if n not in prompt_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay,
                 "lr": self.args.others_lr if self.args.others_lr else self.args.learning_rate}
            ]
            optimizer_cls, optimizer_kwargs = BaseTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

class SoftGlinerTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=8, num_epochs=3, 
                 learning_rate=1e-4, prompt_encoder_lr=5e-4, others_lr=1e-5, freeze_backbone=True):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = {"learning_rate": learning_rate, "prompt_encoder_lr": prompt_encoder_lr,
                       "others_lr": others_lr, "batch_size": batch_size, "num_epochs": num_epochs,
                       "weight_decay": 0.01, "freeze_backbone": freeze_backbone}

    def _freeze_backbone(self):
        """Congela solo token_rep_layer come Stefano, sbloccando rnn, span_rep_layer, etc."""
        if self.config["freeze_backbone"]:
            print("\n" + "="*70)
            print("PARAMETER FREEZING SUMMARY (Stefano-style)")
            print("="*70)
            
            model = self.model.model
            
            # Congela SOLO token_rep_layer (come Stefano)
            components_to_freeze = ['token_rep_layer']
            
            # Prima sblocca tutto
            for param in model.parameters():
                param.requires_grad = True
            
            # Poi congela solo i componenti specificati
            for comp_name in components_to_freeze:
                if hasattr(model, comp_name):
                    comp = getattr(model, comp_name)
                    for param in comp.parameters():
                        param.requires_grad = False
                    print(f"  ❄️ FROZEN: {comp_name}")
            
            # Sblocca il prompt_encoder nel wrapper
            wrapped_encoder = model.token_rep_layer.labels_encoder.model
            if hasattr(wrapped_encoder, 'prompt_encoder'):
                for param in wrapped_encoder.prompt_encoder.parameters():
                    param.requires_grad = True
                print(f"  🔥 UNFROZEN: prompt_encoder (in wrapper)")
            
            # Statistiche per componente (come Stefano)
            print("\n" + "-"*70)
            print("Component-wise Breakdown:")
            print(f"  {'Component':<25} {'Total':<15} {'Trainable':<15} {'% Trainable'}")
            print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
            
            components = {
                'token_rep_layer': getattr(model, 'token_rep_layer', None),
                'rnn': getattr(model, 'rnn', None),
                'span_rep_layer': getattr(model, 'span_rep_layer', None),
                'prompt_rep_layer': getattr(model, 'prompt_rep_layer', None),
                'prompt_encoder': wrapped_encoder.prompt_encoder if hasattr(wrapped_encoder, 'prompt_encoder') else None,
            }
            
            total_params = 0
            trainable_params = 0
            
            for comp_name, comp in components.items():
                if comp is None:
                    continue
                comp_total = sum(p.numel() for p in comp.parameters())
                comp_trainable = sum(p.numel() for p in comp.parameters() if p.requires_grad)
                comp_pct = (comp_trainable / comp_total * 100) if comp_total > 0 else 0
                
                status = "🔥" if comp_trainable > 0 else "❄️"
                print(f"  {status} {comp_name:<23} {comp_total:>13,}  {comp_trainable:>13,}  {comp_pct:>10.2f}%")
                
                total_params += comp_total
                trainable_params += comp_trainable
            
            print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
            print(f"  {'TOTAL':<25} {total_params:>13,}  {trainable_params:>13,}  {trainable_params/total_params*100:>10.2f}%")
            print("="*70 + "\n")

    def train(self):
        self._freeze_backbone()
        training_args = SoftGlinerTrainingArguments(
            output_dir="models_soft_prompt",
            learning_rate=self.config["learning_rate"],
            prompt_encoder_lr=self.config["prompt_encoder_lr"],
            others_lr=self.config["others_lr"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["num_epochs"],
            weight_decay=self.config["weight_decay"],
            eval_strategy="steps", save_strategy="steps", save_total_limit=2,
            load_best_model_at_end=True, use_cpu=not torch.cuda.is_available(), report_to="none"
        )
        import gliner.training
        original_trainer_cls = gliner.training.Trainer
        gliner.training.Trainer = SoftGlinerOptimizer
        try:
            trainer = self.model.train_model(train_dataset=self.train_dataset, eval_dataset=self.val_dataset, training_args=training_args)
        finally:
            gliner.training.Trainer = original_trainer_cls
        return trainer

# ==========================================
# 3. DATA LOADING
# ==========================================
if is_running_on_kaggle():
    MODEL_NAME = '/kaggle/input/glinerbismall2/'
    path = "/kaggle/input/jnlpa-6-2k5-1-2-complete/"
else:
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
    path = "../dataset/"

train_path = path + "dataset_span_bi.json"
val_path = path + "val_dataset_span_bi.json"
test_path = path + "test_dataset_span_bi.json"
label2id_path = path + "label2id.json"

print(f"Loading datasets from {path}...")
with open(train_path, "r") as f: train_dataset = json.load(f)
with open(val_path, "r") as f: val_dataset = json.load(f)
with open(test_path, "r") as f: test_dataset = json.load(f)
with open(label2id_path, "r") as f: label2id = json.load(f)
id2label = {str(v): k for k, v in label2id.items()}

def convert_ids_to_labels(dataset, id_map):
    new_dataset = []
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            label_id = str(label_id)
            if label_id in id_map and id_map[label_id] != "O":
                new_ner.append([start, end, id_map[label_id]])
        item['ner'] = new_ner
        if len(new_ner) > 0: new_dataset.append(item)
    return new_dataset

train_dataset = convert_ids_to_labels(train_dataset, id2label)
val_dataset = convert_ids_to_labels(val_dataset, id2label)
test_dataset = convert_ids_to_labels(test_dataset, id2label)

all_labels = set()
for d in train_dataset + val_dataset:
    for _, _, l in d['ner']: all_labels.add(l)
label_list = sorted(list(all_labels))
print(f"Labels found: {len(label_list)}")

# ==========================================
# 3.5. DEFINIZIONE FUNZIONE METRICHE (prima del training)
# ==========================================
from collections import defaultdict

def calculate_metrics(dataset, model, batch_size=1):
    """Funzione di valutazione robusta dalla v1"""
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
    
    print(f"\nEvaluating on {len(dataset)} samples with {len(label_list)} labels")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    # Process in batches
    debug_counter = 0
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
            
            # 🐛 DEBUG: Stampa primi 3 esempi
            if debug_counter < 3:
                print(f"\n🐛 DEBUG Sample {debug_counter + 1}:")
                print(f"  Text: {' '.join(tokens[:20])}...")
                print(f"  GT Spans: {list(gt_spans)[:5]}")
                print(f"  Raw Preds: {preds[:5]}")
                print(f"  Pred Spans (token): {list(pred_spans)[:5]}")
                debug_counter += 1
            
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
    
    # 🐛 DEBUG: Stampa statistiche
    print(f"\n🐛 DEBUG METRICS:")
    print(f"  Total TP: {sum(tp.values())}")
    print(f"  Total FP: {sum(fp.values())}")
    print(f"  Total FN: {sum(fn.values())}")
    print(f"  Total Support (GT entities): {sum(support.values())}")
    print(f"  Valid Labels: {len(valid_labels)}")
    
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
    
    print("\n## 📈 Global Metrics (Label2ID Mode, EXCLUDING O)\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")
    
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1
    }

# ==========================================
# 4. EXECUTION - Usando wrapper approach (come v1)
# ==========================================
print("INITIALIZING SOFT BI-GLINER (Wrapper Approach)")

# 4.1. Carica modello GLiNER base
model = GLiNER.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"✅ Model loaded on {device}")

# 4.2. Verifica che sia bi-encoder
if not hasattr(model.model, 'token_rep_layer') or not hasattr(model.model.token_rep_layer, 'labels_encoder'):
    raise ValueError("Il modello non è un Bi-Encoder!")

# 4.3. Ottieni dimensioni e crea PromptEncoder
lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
original_embeddings = lbl_enc_model.embeddings.word_embeddings
embed_dim = original_embeddings.embedding_dim
print(f"✅ Label Embedding Dim: {embed_dim}")

# 4.4. Crea PromptEncoder
prompt_encoder = PromptEncoder(
    backbone_hidden_size=embed_dim,
    num_labels=len(label_list)
).to(device)
print(f"✅ PromptEncoder creato con dim={embed_dim}, num_labels={len(label_list)}")

# 4.5. Crea wrapper e inietta nel modello
wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)
model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
print("✅ SoftPromptLabelEncoderWrapper iniettato nel modello!")

# 4.6. Verifica device
print(f"🖥️ Device: {device}")
print(f"🖥️ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 4.5. BASELINE TEST (PRE-TRAINING) - Usando GLiNER evaluate()
# ==========================================
print("\n" + "="*50)
print("🔬 BASELINE EVALUATION (Pre-Training)")
print("="*50)
model.eval()
baseline_output, baseline_f1 = model.evaluate(
    test_data=test_dataset,
    flat_ner=True,
    multi_label=False,
    threshold=0.5,
    batch_size=8
)
print(f"Baseline Results:\n{baseline_output}")
print(f"Baseline F1: {baseline_f1:.4f}")
print("="*50)

trainer_wrapper = SoftGlinerTrainer(
    model=model, train_dataset=train_dataset, val_dataset=val_dataset,
    batch_size=8, num_epochs=5, learning_rate=1e-4, prompt_encoder_lr=5e-4, others_lr=1e-5, freeze_backbone=True
)

print("Starting training...")
# Nota: load_best_model_at_end=True nel trainer assicura che alla fine 
# 'model' contenga i pesi della miglior epoca di validazione.
trainer_wrapper.train()

# ==========================================
# 5. SALVATAGGIO SELETTIVO (SOLO PROMPT ENCODER)
# ==========================================
from typing import Set
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(f"best_soft_model_{timestamp}")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"\n💾 Salvataggio del miglior Prompt Encoder in {save_dir}...")

# 1. Salva solo lo state_dict del Prompt Encoder (super leggero) con TIMESTAMP
prompt_encoder_filename = f"prompt_encoder_{timestamp}.pt"
# Con l'approccio wrapper, il prompt_encoder è nel wrapped_encoder
wrapped_encoder = model.model.token_rep_layer.labels_encoder.model
torch.save(wrapped_encoder.prompt_encoder.state_dict(), save_dir / prompt_encoder_filename)

# 2. Salva Iperparametri
metadata = {
    "hyperparameters": trainer_wrapper.config,
    "dataset_info": {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "labels": label_list
    },
    "backbone": MODEL_NAME,
    "timestamp": timestamp,
    "model_filename": prompt_encoder_filename
}

with open(save_dir / f"metadata_prompt_tuning_{timestamp}.json", "w") as f:
    json.dump(metadata, f, indent=4)
    
print("✅ Salvataggio completato.")

# ==========================================
# 6. TESTING FINALE (POST-TRAINING) - Usando GLiNER evaluate() come in validazione
# ==========================================

print("\n" + "="*50)
print("🧪 FINAL EVALUATION (Post-Training)")
print("="*50)
model.eval()

# Usa lo stesso metodo di valutazione usato durante il training!
output_str, f1_score = model.evaluate(
    test_data=test_dataset,
    flat_ner=True,
    multi_label=False,
    threshold=0.5,
    batch_size=8
)

print(f"\n📊 Risultati Test Set:")
print(output_str)
print(f"F1 Score: {f1_score:.4f}")

# Salva metriche
import re
pattern = r'P:\s*([\d.]+)%\s*R:\s*([\d.]+)%\s*F1:\s*([\d.]+)%'
match = re.search(pattern, output_str)

if match:
    metrics = {
        "precision": float(match.group(1)) / 100.0,
        "recall": float(match.group(2)) / 100.0,
        "f1": float(match.group(3)) / 100.0
    }
else:
    metrics = {"f1": f1_score, "raw_output": output_str}

metrics_filename = f"test_metrics_{timestamp}.json"
with open(save_dir / metrics_filename, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n✅ Script terminato.")
print(f"📁 Directory Output: {save_dir}")
print(f"📄 Metriche salvate in: {metrics_filename}")