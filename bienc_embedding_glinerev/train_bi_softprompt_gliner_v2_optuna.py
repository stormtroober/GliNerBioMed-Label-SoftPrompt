# -*- coding: utf-8 -*-
"""
Training Bi-Encoder Soft Prompt GLiNER - OPTUNA HYPERPARAMETER SEARCH v2
- Obiettivo: Trovare i migliori iperparametri di training per soft prompt GLiNER
- Architettura Fissata: PromptEncoder (Linear + LayerNorm)
- Parametri Ottimizzati:
  * prompt_encoder_lr (1e-4 to 1e-2, log scale)
  * others_lr (1e-6 to 1e-3, log scale)
  * warmup_ratio (0.0 to 0.2)
  * use_focal (True/False) - test con e senza focal loss
  * focal_loss_gamma (2.0 to 5.0) - solo se use_focal=True
  * focal_loss_alpha (0.25 to 0.75) - solo se use_focal=True

CONFIGURAZIONE:
- 15 trials (Budget: ~2.5h con 2min/epoca, ~2h con pruning)
- 5 epoche per trial
- MedianPruner per early stopping dei trial cattivi
  
CHANGELOG v2:
- Aggiunto warmup_ratio come parametro ottimizzabile (era fisso a 0.1)
- Modificato range focal_loss_gamma a [2.0-5.0] (richiesta utente)
- Aggiunto parametro categorico use_focal per testare anche senza focal loss
- Ridotto NUM_TRIALS a 15 per rispettare budget di 2-3 ore
- Migliorate le stampe di debug per includere tutti i parametri
"""



import os
import json
import random
import logging
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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

# ==========================================================
# 🔧 CONFIGURAZIONE BASE (Fissa)
# ==========================================================
BATCH_SIZE = 8
EPOCHS = 5  # Epoche per trial (2 min/epoca = 10 min/trial)
NUM_TRIALS = 20
FREEZE_BACKBONE = True
LR_SCHEDULER_TYPE = "linear"  # Scheduler fisso, warmup_ratio sarà ottimizzato

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
# 1.5. LABEL ENCODER WRAPPER
# ==========================================

class SoftPromptLabelEncoderWrapper(nn.Module):
    """Wrapper che intercetta le chiamate al label encoder e inietta soft prompts."""
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
        if inputs_embeds is None and input_ids is not None:
            raw_embeds = self.original_encoder.embeddings(input_ids)
            position_mask = self._create_position_mask(input_ids)
            inputs_embeds = self.prompt_encoder(raw_embeds, position_mask)
            
            if not self._debug_printed:
                print(f"\n[WRAPPER DEBUG]")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Raw embeds shape: {raw_embeds.shape}")
                print(f"  Soft embeds shape: {inputs_embeds.shape}")
                self._debug_printed = True
        
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

from dataclasses import dataclass
from typing import Optional

@dataclass
class SoftGlinerTrainingArguments(BaseTrainingArguments):
    prompt_encoder_lr: Optional[float] = None
    focal_loss_alpha: Optional[float] = None
    focal_loss_gamma: Optional[float] = None

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
                 learning_rate=1e-4, prompt_encoder_lr=5e-4, others_lr=1e-5, freeze_backbone=True,
                 focal_loss_alpha=0.75, focal_loss_gamma=2, lr_scheduler_type="linear", warmup_ratio=0.1):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = {"learning_rate": learning_rate, "prompt_encoder_lr": prompt_encoder_lr,
                       "others_lr": others_lr, "batch_size": batch_size, "num_epochs": num_epochs,
                       "weight_decay": 0.01, "freeze_backbone": freeze_backbone,
                       "focal_loss_alpha": focal_loss_alpha, "focal_loss_gamma": focal_loss_gamma,
                       "lr_scheduler_type": lr_scheduler_type, "warmup_ratio": warmup_ratio}

    def _freeze_backbone(self):
        """Congela solo token_rep_layer, sbloccando rnn, span_rep_layer, etc."""
        if self.config["freeze_backbone"]:
            model = self.model.model
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
            
            # Sblocca il prompt_encoder nel wrapper
            wrapped_encoder = model.token_rep_layer.labels_encoder.model
            if hasattr(wrapped_encoder, 'prompt_encoder'):
                for param in wrapped_encoder.prompt_encoder.parameters():
                    param.requires_grad = True

    def train(self, silent=False):
        self._freeze_backbone()
        
        # Calcola steps per epoca per logging
        num_samples = len(self.train_dataset)
        steps_per_epoch = num_samples // self.config["batch_size"]
        logging_steps = max(1, steps_per_epoch // 4)
        save_steps = steps_per_epoch
        
        if not silent:
            print(f"\n⏱️ TRAINING TIMING INFO:")
            print(f"  Samples: {num_samples}")
            print(f"  Batch size: {self.config['batch_size']}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total epochs: {self.config['num_epochs']}\n")
        
        # Detect CUDA/bf16/fp16
        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = use_cuda and not use_bf16
        
        training_args = SoftGlinerTrainingArguments(
            output_dir="models_soft_prompt_optuna",
            learning_rate=self.config["learning_rate"],
            prompt_encoder_lr=self.config["prompt_encoder_lr"],
            others_lr=self.config["others_lr"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["num_epochs"],
            weight_decay=self.config["weight_decay"],
            focal_loss_alpha=self.config["focal_loss_alpha"],
            focal_loss_gamma=self.config["focal_loss_gamma"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            warmup_ratio=self.config["warmup_ratio"],
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            use_cpu=not use_cuda,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none"
        )
        
        import gliner.training
        original_trainer_cls = gliner.training.Trainer
        gliner.training.Trainer = SoftGlinerOptimizer
        
        start_time = time.time()
        
        try:
            trainer = self.model.train_model(train_dataset=self.train_dataset, eval_dataset=self.val_dataset, training_args=training_args)
        finally:
            gliner.training.Trainer = original_trainer_cls
        
        total_time = time.time() - start_time
        
        if not silent:
            print(f"\n⏱️ TRAINING COMPLETE:")
            print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)\n")
        
        # Estrai la miglior val loss dal trainer
        best_val_loss = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None
        
        return trainer, best_val_loss

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
# 🎯 OPTUNA OBJECTIVE
# ==========================================
def objective(trial):
    """Optuna objective per ottimizzare iperparametri di training."""
    
    # --- 1. SUGGERISCI IPERPARAMETRI ---
    prompt_encoder_lr = trial.suggest_float("prompt_encoder_lr", 1e-4, 1e-2, log=True)
    others_lr = trial.suggest_float("others_lr", 1e-6, 1e-3, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Focal Loss: testa anche configurazione senza focal loss
    use_focal = trial.suggest_categorical("use_focal", [True, False])
    if use_focal:
        focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 2.0, 5.0)  # Range moderato
        focal_loss_alpha = trial.suggest_float("focal_loss_alpha", 0.25, 0.75)
    else:
        # Valori dummy quando focal loss è disabilitata (GLiNER userà CE standard)
        focal_loss_gamma = None
        focal_loss_alpha = None
    
    print(f"\n🧪 TRIAL {trial.number}")
    print(f"  LR_PromptEncoder: {prompt_encoder_lr:.2e}")
    print(f"  LR_Others: {others_lr:.2e}")
    print(f"  Warmup Ratio: {warmup_ratio:.2f}")
    print(f"  Use Focal Loss: {use_focal}")
    if use_focal:
        print(f"  Focal Gamma: {focal_loss_gamma:.2f}")
        print(f"  Focal Alpha: {focal_loss_alpha:.2f}")
    
    # --- 2. CARICA MODELLO FRESCO ---
    model = GLiNER.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Verifica bi-encoder
    if not hasattr(model.model, 'token_rep_layer') or not hasattr(model.model.token_rep_layer, 'labels_encoder'):
        raise ValueError("Il modello non è un Bi-Encoder!")
    
    # Ottieni dimensioni e crea PromptEncoder
    lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    embed_dim = original_embeddings.embedding_dim
    
    prompt_encoder = PromptEncoder(
        backbone_hidden_size=embed_dim,
        num_labels=len(label_list)
    ).to(device)
    
    # Crea wrapper e inietta nel modello
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)
    model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
    
    # --- 3. TRAINING ---
    trainer_wrapper = SoftGlinerTrainer(
        model=model, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE, 
        num_epochs=EPOCHS, 
        learning_rate=1e-4,  # Base LR (usato come fallback)
        prompt_encoder_lr=prompt_encoder_lr, 
        others_lr=others_lr, 
        freeze_backbone=FREEZE_BACKBONE, 
        focal_loss_alpha=focal_loss_alpha if use_focal else 0.25,  # Default se non focal
        focal_loss_gamma=focal_loss_gamma if use_focal else 0.0,  # 0.0 = CE standard
        lr_scheduler_type=LR_SCHEDULER_TYPE, 
        warmup_ratio=warmup_ratio  # Ora ottimizzato!
    )
    
    epoch_start = time.time()
    trainer, best_val_loss = trainer_wrapper.train(silent=True)
    epoch_duration = time.time() - epoch_start
    
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Trial Duration: {epoch_duration:.2f}s ({epoch_duration/60:.2f}m)")
    
    # Cleanup
    del model
    del prompt_encoder
    del wrapped_encoder
    del trainer
    del trainer_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_val_loss

# ==========================================================
# 🚀 MAIN
# ==========================================================
if __name__ == "__main__":
    print(f"\n🏗️ Inizio Soft Prompt GLiNER Hyperparam Search ({NUM_TRIALS} Trials, {EPOCHS} Epoche per trial)")
    print(f"🔒 Fixed Arch: PromptEncoder (Linear + LayerNorm)")
    
    # Pruning aggiustato per 5 epoche
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print("\n🏆 BEST PARAMS:")
    print(study.best_params)
    print(f"Best Val Loss: {study.best_value}")

    # SAVE RESULTS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("softprompting/optunas/gliner_v2_hyper")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON Results
    results = {
        "timestamp": timestamp,
        "fixed_config": {
            "batch_size": BATCH_SIZE,
            "epochs_per_trial": EPOCHS,
            "num_trials": NUM_TRIALS,
            "freeze_backbone": FREEZE_BACKBONE,
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "architecture": "PromptEncoder (Linear + LayerNorm)"
        },
        "optimized_params": [
            "prompt_encoder_lr (1e-4 to 1e-2, log)",
            "others_lr (1e-6 to 1e-3, log)",
            "warmup_ratio (0.0 to 0.2)",
            "use_focal (True/False)",
            "focal_loss_gamma (2.0 to 5.0, if focal)",
            "focal_loss_alpha (0.25 to 0.75, if focal)"
        ],
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": []
    }
    
    for t in study.trials:
        results["trials"].append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state)
        })
        
    json_path = output_dir / f"hyper_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {json_path}")

    # PLOTS
    try:
        # Importanza Iperparametri
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title("Soft Prompt GLiNER Hyperparameter Importance")
        plt.tight_layout()
        importance_path = output_dir / f"hyper_importance_{timestamp}.png"
        plt.savefig(importance_path)
        plt.close()
        print(f"📊 Importance plot saved to: {importance_path}")

        # Slice Plot
        fig = optuna.visualization.matplotlib.plot_slice(study)
        plt.tight_layout()
        slice_path = output_dir / f"hyper_slice_{timestamp}.png"
        plt.savefig(slice_path)
        plt.close()
        print(f"📊 Slice plot saved to: {slice_path}")
        
    except Exception as e:
        print(f"⚠️ Errore plotting avanzato: {e}")
        # Fallback manual plot
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if complete_trials:
            lrs = [t.params['prompt_encoder_lr'] for t in complete_trials]
            vals = [t.value for t in complete_trials]
            plt.figure()
            plt.scatter(lrs, vals)
            plt.xscale('log')
            plt.xlabel('Prompt Encoder Learning Rate')
            plt.ylabel('Val Loss')
            plt.title('LR Prompt Encoder Impact')
            manual_path = output_dir / f"manual_lr_plot_{timestamp}.png"
            plt.savefig(manual_path)
            plt.close()
            print(f"📊 Manual plot saved to: {manual_path}")

    print(f"\n✅ Optuna search completato!")
    print(f"📁 Output directory: {output_dir}")
