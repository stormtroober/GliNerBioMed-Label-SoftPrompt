
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json 
import random
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from gliner import GLiNER
import shutil
import optuna
import datetime
from optuna.trial import TrialState

# ==========================================
# CONFIGURATION
# ==========================================

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

if is_running_on_kaggle():
    path = "/kaggle/input/jnlpa-6-2k5-1-2-complete/"
    MODEL_NAME = '/kaggle/input/glinerbismall2/'
else:
    path = "../dataset/"
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

train_path = path + "dataset_span_bi.json"
val_path = path + "val_dataset_span_bi.json"
test_path = path + "test_dataset_span_bi.json"
label2id_path = path + "label2id.json"

# Global settings for trials
TARGET_EPOCHS = 3 
BATCH_SIZE = 8

# ==========================================
# CLASSES (Prompt Encoder)
# ==========================================

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, hidden_dim=None, dropout=0.1, tokenizer=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        if hidden_dim is None: hidden_dim = embed_dim 
            
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
        # 1. Get original embeddings
        x = self.embedding(input_ids)
        
        # 2. Compute Transform
        delta = self.mlp(x)
        
        # 3. Apply Mask: Do NOT transform [CLS](101), [SEP](102), [PAD](0)
        mask = (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
        mask_float = mask.unsqueeze(-1).expand_as(delta).float()
        
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
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_encoder, name)



# ==========================================
# METRICS FUNCTION (Modified to return F1)
# ==========================================
def calculate_metrics_for_optuna(dataset, model, batch_size=1):
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Gather all labels
    all_labels = set()
    for d in dataset:
        for x in d['ner']: 
            all_labels.add(x[2])
    if 'O' in all_labels:
        all_labels.remove('O')
        
    label_list = sorted(list(all_labels))
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
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
            gt_spans = set()
            for s, e, l in item['ner']:
                if l != 'O': 
                    gt_spans.add((l, s, e)) 
                    support[l] += 1
            
            preds = batch_preds[idx]
            
            tokens = item['tokenized_text']
            char_to_token = {}
            cursor = 0
            for t_i, token in enumerate(tokens):
                start = cursor
                end = cursor + len(token)
                for c in range(start, end):
                    char_to_token[c] = t_i
                cursor = end + 1 
            
            pred_spans = set()
            for p in preds:
                label = p['label']
                if label == 'O': continue 

                p_start = p['start']
                p_end = p['end'] 
                
                if p_start in char_to_token and (p_end - 1) in char_to_token:
                    t_start = char_to_token[p_start]
                    t_end = char_to_token[p_end - 1]
                    pred_spans.add((label, t_start, t_end))
            
            tps = pred_spans.intersection(gt_spans)
            fps = pred_spans - gt_spans
            fns = gt_spans - pred_spans
            
            for l, s, e in tps: tp[l] += 1
            for l, s, e in fps: fp[l] += 1
            for l, s, e in fns: fn[l] += 1

    # Global Metrics
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
        
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    return macro_f1

# ==========================================
# DATA LOADING
# ==========================================

print("Loading datasets for Optuna study...")
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
                if label_name == "O": continue
                new_ner.append([start, end, label_name])
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset

train_dataset = convert_ids_to_labels(train_dataset, id2label)
val_dataset = convert_ids_to_labels(val_dataset, id2label)
test_dataset = convert_ids_to_labels(test_dataset, id2label)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================

def objective(trial):
    # 1. Suggest Hyperparameters (REDUCED for 8h budget)
    # Optimized: 3 key parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 2.0, 4.0)
    
    # Fixed parameters (from train_bi_softprompt_gliner.py)
    others_lr = 5e-6                # Same as original
    weight_decay = 0.01             # Same as original
    others_weight_decay = 0.05      # Same as original
    
    # 2. Setup Model (Fresh Copy)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained(MODEL_NAME).to(device)
    
    # 3. Inject Soft Prompt Encoder with trial settings
    lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    vocab_size = original_embeddings.num_embeddings
    embed_dim = original_embeddings.embedding_dim
    
    prompt_encoder = MLPPromptEncoder(
        original_embeddings, 
        vocab_size, 
        embed_dim, 
        dropout=dropout,
        tokenizer=None # Skip tokenizer debug for speed
    ).to(device)
    
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)
    model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
    
    for p in model.parameters():
        p.requires_grad = True

    # 4. Training EPOCH BY EPOCH (per pruning)
    data_size = len(train_dataset)
    num_batches_per_epoch = data_size // BATCH_SIZE
    
    # Mixed precision settings (come in train_bi_softprompt_gliner.py)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    
    print(f"\n[Trial {trial.number}] 📍 Training started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"[Trial {trial.number}] Steps per epoch: {num_batches_per_epoch}, Total epochs: {TARGET_EPOCHS}")
    
    trial_start_time = time.time()
    best_val_f1 = 0.0
    
    for epoch in range(TARGET_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n[Trial {trial.number}] 📍 EPOCH {epoch + 1}/{TARGET_EPOCHS} started at {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Train for 1 epoch
        model.train_model(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=f"optuna_models/trial_{trial.number}",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            others_lr=others_lr,
            others_weight_decay=others_weight_decay,
            lr_scheduler_type="linear",
            warmup_ratio=0.1 if epoch == 0 else 0.0,  # Warmup only on first epoch
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            focal_loss_alpha=0.75,
            focal_loss_gamma=focal_loss_gamma,
            num_train_epochs=1,  # Train 1 epoch at a time
            save_steps=num_batches_per_epoch,
            logging_steps=num_batches_per_epoch,
            save_total_limit=1,
            eval_strategy="no",
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            use_cpu=not use_cuda,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
        )
        
        epoch_elapsed = time.time() - epoch_start_time
        
        # Evaluate after each epoch
        macro_f1_val = calculate_metrics_for_optuna(val_dataset, model, batch_size=BATCH_SIZE)
        
        print(f"[Trial {trial.number}] ✅ EPOCH {epoch + 1} finished - Elapsed: {epoch_elapsed:.2f}s ({epoch_elapsed/60:.2f}min)")
        print(f"[Trial {trial.number}] Validation Macro F1: {macro_f1_val:.4f}")
        
        # Track best
        if macro_f1_val > best_val_f1:
            best_val_f1 = macro_f1_val
        
        # Report intermediate value to Optuna
        trial.report(macro_f1_val, epoch)
        
        # Check if trial should be pruned (after epoch 2, i.e., epoch index 1)
        if trial.should_prune():
            print(f"[Trial {trial.number}] ✂️ PRUNED after epoch {epoch + 1}")
            # Cleanup before pruning
            del model
            del prompt_encoder
            del wrapped_encoder
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
    
    trial_elapsed = time.time() - trial_start_time
    print(f"\n[Trial {trial.number}] ✅ Training COMPLETED at {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"[Trial {trial.number}] Total training time: {trial_elapsed:.2f}s ({trial_elapsed/60:.2f}min)")
    
    # 5. Final Evaluation on Test (Information only)
    macro_f1_test = calculate_metrics_for_optuna(test_dataset, model, batch_size=BATCH_SIZE)
    print(f"[Trial {trial.number}] Final Validation F1: {best_val_f1:.4f}, Test F1: {macro_f1_test:.4f}")
    
    # Cleanup
    del model
    del prompt_encoder
    del wrapped_encoder
    torch.cuda.empty_cache()
    
    return best_val_f1

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Starting Optuna Study with Pruning...")
    study_name = "softprompt_optimization"
    
    # Create study with MedianPruner
    # n_warmup_steps=1 means pruning can happen after epoch 2 (step index 1)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,      # Wait for 3 trials before pruning
        n_warmup_steps=1,        # Start pruning after epoch 2 (0-indexed: step 1)
        interval_steps=1         # Check at every epoch
    )
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        pruner=pruner
    )
    
    # Optimize (5 trials fit in ~8h budget with 3 epochs each @ ~104min/trial)
    study.optimize(objective, n_trials=5) 
    
    print("\n" + "="*50)
    print("OPTUNA RESULTS")
    print("="*50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save Best Parameters
    os.makedirs("optuna_results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"optuna_results/optuna_best_params_{timestamp}.json", "w") as f:
        json.dump(trial.params, f, indent=4)
        
    # ==========================================================
    # 📊 VISUALIZZAZIONE
    # ==========================================================
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Statistiche trial
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        
        # History plot
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        
        axes[0].scatter(trial_numbers, trial_values, alpha=0.7)
        # Max so far because we are maximizing F1
        axes[0].plot(trial_numbers, [max(trial_values[:i+1]) for i in range(len(trial_values))], 
                     'r-', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Macro F1 Score')
        axes[0].set_title('Optimization History (F1 Maximization)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Parameter importance
        params_names = list(trial.params.keys())
        importance_values = []
        for p in params_names:
            try:
                imp = optuna.importance.get_param_importances(study)[p]
            except:
                imp = 0
            importance_values.append(imp)
        
        # Sort for better visualization
        sorted_idx = np.argsort(importance_values)
        sorted_names = [params_names[i] for i in sorted_idx]
        sorted_vals = [importance_values[i] for i in sorted_idx]
        
        axes[1].barh(sorted_names, sorted_vals, color='steelblue')
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
