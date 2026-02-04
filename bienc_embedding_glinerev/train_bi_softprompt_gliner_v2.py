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
from gliner.modeling.base import BiEncoderSpanModel
from gliner.modeling.utils import extract_word_embeddings
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

class CustomBiEncoderSpanModel(BiEncoderSpanModel):
    def __init__(self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None, 
                 prompt_encoder: Optional[Any] = None, fixed_labels: Optional[list] = None) -> None:
        super().__init__(config, from_pretrained, cache_dir)
        if prompt_encoder is not None:
            self.prompt_encoder = prompt_encoder
        self.fixed_labels = fixed_labels
        self._debug_printed = False
        
        # --- CACCIA AL TESORO: TROVA IL LAYER DI PROIEZIONE ---
        self.projection_layer = None
        
        # Tentativo 1: Nome standard
        if hasattr(self.token_rep_layer, 'projection'):
            self.projection_layer = self.token_rep_layer.projection
            print("✅ Projection Layer trovato (metodo standard).")
        else:
            # Tentativo 2: Cerca tra i moduli figli un Linear 384->768
            print("⚠️ Projection Layer 'projection' non trovato. Cerco manualmente...")
            found = False
            for name, module in self.token_rep_layer.named_children():
                if isinstance(module, nn.Linear):
                    # Verifica le dimensioni
                    if module.in_features == 384 and module.out_features == 768:
                        self.projection_layer = module
                        print(f"✅ Projection Layer trovato: '{name}' ({module})")
                        found = True
                        break
            
            if not found:
                print("❌ CRITICAL: Nessun Projection Layer 384->768 trovato! Il modello crasherà.")

    def _create_position_mask_for_labels(self, label_input_ids: torch.Tensor) -> torch.Tensor:
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 101, 102, 0
        position_mask = torch.zeros_like(label_input_ids, dtype=torch.long)
        mask = (label_input_ids != BOS_TOKEN) & (label_input_ids != EOS_TOKEN) & (label_input_ids != PAD_TOKEN)
        position_mask[mask] = 1
        return position_mask

    def get_representations(
        self, input_ids=None, attention_mask=None, labels_embeds=None, labels_input_ids=None, 
        labels_attention_mask=None, text_lengths=None, words_mask=None, **kwargs
    ):
        # 1. Text Encode
        token_embeds = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)

        # 2. Label Encode
        labels_encoder = self.token_rep_layer.labels_encoder
        labels_position_mask = self._create_position_mask_for_labels(labels_input_ids)
        
        # 2b. Raw Embeddings (384)
        raw_labels_embeds = labels_encoder.model.embeddings(labels_input_ids) 

        # 2c. Soft Prompt (384 -> 384)
        if hasattr(self, 'prompt_encoder'):
            labels_embeds = self.prompt_encoder(raw_labels_embeds, labels_position_mask)
        else:
            labels_embeds = raw_labels_embeds

        # 2d. Transformer (384 -> 384)
        labels_out = labels_encoder.model(inputs_embeds=labels_embeds)
        labels_embeds = labels_out.last_hidden_state[:, 0, :] # Pooling [CLS]
        
        # 2e. PROJECTION (384 -> 768) - USIAMO QUELLO TROVATO IN INIT
        if self.projection_layer is not None:
             labels_embeds = self.projection_layer(labels_embeds)

        # DEBUG PRINTS
        if not self._debug_printed:
            print(f"\n[DEBUG SHAPES]")
            print(f"Raw Label Emb: {raw_labels_embeds.shape}")
            print(f"Projected Label: {labels_embeds.shape} (TARGET: 768)")
            print(f"Text Emb: {token_embeds.shape}")
            self._debug_printed = True

        # 3. Output packaging
        batch_size, _, embed_dim = token_embeds.shape
        max_text_length = text_lengths.max()
        words_embedding, mask = extract_word_embeddings(
            token_embeds, words_mask, attention_mask, batch_size, max_text_length, embed_dim, text_lengths
        )
        labels_embeds = labels_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        labels_embeds = labels_embeds.to(words_embedding.dtype)

        return labels_embeds, labels_mask, words_embedding, mask

class SoftBiGliner(GLiNER):
    @classmethod
    def from_pretrained(cls, model_id, prompt_encoder_cls=None, fixed_labels=None, **kwargs):
        instance = super().from_pretrained(model_id, **kwargs)
        if not hasattr(instance.model, 'token_rep_layer') or not hasattr(instance.model.token_rep_layer, 'labels_encoder'):
            raise ValueError("Il modello non è un Bi-Encoder.")

        # RILEVAMENTO DIMENSIONE REALE (384)
        try:
            label_emb_layer = instance.model.token_rep_layer.labels_encoder.model.embeddings.word_embeddings
            label_embedding_dim = label_emb_layer.weight.shape[1]
            print(f"✅ Label Dim: {label_embedding_dim}")
        except AttributeError:
            label_embedding_dim = instance.model.token_rep_layer.labels_encoder.config.hidden_size
            print(f"⚠️ Config Dim: {label_embedding_dim}")

        prompt_encoder = None
        if prompt_encoder_cls is not None and fixed_labels is not None:
            # INIZIALIZZAZIONE CORRETTA A 384
            prompt_encoder = prompt_encoder_cls(backbone_hidden_size=label_embedding_dim, num_labels=len(fixed_labels))
            print(f"✅ PromptEncoder inizializzato a dim: {label_embedding_dim}")

        old_model = instance.model
        instance.model = CustomBiEncoderSpanModel(old_model.config, prompt_encoder=prompt_encoder, fixed_labels=fixed_labels)
        instance.model.load_state_dict(old_model.state_dict(), strict=False)
        instance.model.to(instance.device)
        return instance

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
                 learning_rate=5e-5, prompt_encoder_lr=1e-4, others_lr=1e-6, freeze_backbone=True):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = {"learning_rate": learning_rate, "prompt_encoder_lr": prompt_encoder_lr,
                       "others_lr": others_lr, "batch_size": batch_size, "num_epochs": num_epochs,
                       "weight_decay": 0.01, "freeze_backbone": freeze_backbone}

    def _freeze_backbone(self):
        if self.config["freeze_backbone"]:
            print("\n❄️ FREEZING BACKBONE...")
            for param in self.model.model.parameters(): param.requires_grad = False
            if hasattr(self.model.model, 'prompt_encoder'):
                for param in self.model.model.prompt_encoder.parameters(): param.requires_grad = True
                print("🔥 Prompt Encoder: UNFROZEN")
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable Params: {trainable:,}")

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
# 4. EXECUTION
# ==========================================
print("INITIALIZING SOFT BI-GLINER")
model = SoftBiGliner.from_pretrained(MODEL_NAME, prompt_encoder_cls=PromptEncoder, fixed_labels=label_list)

trainer_wrapper = SoftGlinerTrainer(
    model=model, train_dataset=train_dataset, val_dataset=val_dataset,
    batch_size=8, num_epochs=1, learning_rate=1e-4, prompt_encoder_lr=5e-4, others_lr=1e-6, freeze_backbone=True
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
torch.save(model.model.prompt_encoder.state_dict(), save_dir / prompt_encoder_filename)

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
# 6. TESTING E CALCOLO METRICHE (AGGIORNATO)
# ==========================================
print("\n🧪 Esecuzione Test Set con conversione Caratteri -> Token...")

model.eval()

def convert_char_to_token_spans(char_entities: List[Dict], tokens: List[str]) -> List[Tuple[int, int, str]]:
    """Helper per convertire output GLiNER (caratteri) in formato Dataset (token)"""
    char_to_token = []
    
    for token_idx, token in enumerate(tokens):
        for _ in range(len(token)):
            char_to_token.append(token_idx)
        if token_idx < len(tokens) - 1:
            char_to_token.append(token_idx)
            
    token_spans = []
    for entity in char_entities:
        start_char = entity['start']
        end_char = entity['end'] - 1 
        label = entity['label']

        if start_char < len(char_to_token) and end_char < len(char_to_token):
            start_token = char_to_token[start_char]
            end_token = char_to_token[end_char]
            token_spans.append((start_token, end_token, label))
            
    return token_spans

def evaluate_token_level(model, dataset, labels):
    results = []
    for sample in tqdm(dataset, desc="Valutazione Test Set"):
        text = " ".join(sample['tokenized_text'])
        preds_char = model.predict_entities(text, labels=labels, threshold=0.5, flat_ner=True)
        preds_token = convert_char_to_token_spans(preds_char, sample['tokenized_text'])
        
        results.append({
            "text": text,
            "truth": sample['ner'], # [start, end, label]
            "pred": preds_token
        })
    return results

def compute_span_metrics(results: List[Dict], ignore_labels: Set[str] = None):
    """Calcola Micro e Macro F1 ignorando specifiche etichette (es. 'O')"""
    if ignore_labels is None:
        ignore_labels = {"O"}

    # Contatori Globali (Micro)
    g_tp, g_fp, g_fn = 0, 0, 0
    # Contatori per Classe (Macro)
    class_stats = {}

    for item in results:
        # Convertiamo in set di tuple (start, end, label) filtrando 'O'
        true_spans = set(tuple(x) for x in item['truth'] if x[2] not in ignore_labels)
        pred_spans = set(tuple(x) for x in item['pred'] if x[2] not in ignore_labels)

        # Calcolo intersezioni
        tp_spans = true_spans & pred_spans
        fp_spans = pred_spans - true_spans
        fn_spans = true_spans - pred_spans

        # Aggiornamento Micro
        g_tp += len(tp_spans)
        g_fp += len(fp_spans)
        g_fn += len(fn_spans)

        # --- CORREZIONE QUI SOTTO ---
        # Estraiamo le classi uniche presenti in questo documento
        # Usiamo le parentesi graffe {} per creare direttamente dei SET, non liste []
        true_classes = {x[2] for x in true_spans}
        pred_classes = {x[2] for x in pred_spans}
        
        # Ora possiamo usare l'operatore | perché sono entrambi set
        all_classes = true_classes | pred_classes
        # ----------------------------

        for cls in all_classes:
            if cls not in class_stats: class_stats[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[cls]['tp'] += sum(1 for x in tp_spans if x[2] == cls)
            class_stats[cls]['fp'] += sum(1 for x in fp_spans if x[2] == cls)
            class_stats[cls]['fn'] += sum(1 for x in fn_spans if x[2] == cls)

    # Calcolo Micro Metrics
    micro_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
    micro_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Calcolo Macro Metrics
    macro_f1_scores = []
    for cls, stats in class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        macro_f1_scores.append(f1)
    
    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0

    return {
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "global_tp": g_tp,
        "global_fp": g_fp,
        "global_fn": g_fn
    }

# 1. Esegui Predizioni
test_results = evaluate_token_level(model, test_dataset, label_list)

# 2. Calcola Metriche
metrics = compute_span_metrics(test_results, ignore_labels={"O"})

# 3. Stampa Report
print("\n" + "="*30)
print(f"📊 RISULTATI TEST SET (No 'O')")
print("="*30)
print(f"Micro Precision: {metrics['micro_precision']}")
print(f"Micro Recall:    {metrics['micro_recall']}")
print(f"Micro F1:        {metrics['micro_f1']}")
print("-" * 20)
print(f"Macro F1:        {metrics['macro_f1']}")
print("="*30)

# 4. Salva Risultati Completi e Metriche
results_filename = f"test_results_{timestamp}.json"
metrics_filename = f"test_metrics_{timestamp}.json"

with open(save_dir / results_filename, "w") as f:
    json.dump(test_results, f, indent=2)

with open(save_dir / metrics_filename, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n✅ Script terminato.")
print(f"📁 Directory Output: {save_dir}")
print(f"📄 Metriche salvate in: {metrics_filename}")
print(f"📄 Risultati dettagliati in: {results_filename}")