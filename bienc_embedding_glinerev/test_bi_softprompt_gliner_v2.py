
import os
import json
import torch
import torch.nn as nn
from gliner import GLiNER
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ==========================================
# UTIL & CLASSES
# ==========================================

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

class PromptEncoder(nn.Module):
    """Same architecture as in train_bi_softprompt_gliner_v2.py"""
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


class SoftPromptLabelEncoderWrapper(nn.Module):
    """Wrapper that intercepts label encoder calls and injects soft prompts.
    
    This approach keeps the GLiNER model intact and only modifies
    the embedding injection point.
    """
    def __init__(self, original_encoder, prompt_encoder, tokenizer=None):
        super().__init__()
        self.original_encoder = original_encoder
        self.prompt_encoder = prompt_encoder
        self.tokenizer = tokenizer
        self._debug_printed = False
        
    def _create_position_mask(self, input_ids):
        """Create mask to identify label tokens (excluding special tokens)."""
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 101, 102, 0
        mask = (input_ids != BOS_TOKEN) & (input_ids != EOS_TOKEN) & (input_ids != PAD_TOKEN)
        return mask.long()
        
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        # If we already have embeddings, use those
        if inputs_embeds is None and input_ids is not None:
            # 1. Get original embeddings
            raw_embeds = self.original_encoder.embeddings(input_ids)
            
            # 2. Create position mask
            position_mask = self._create_position_mask(input_ids)
            
            # 3. Apply soft prompt
            inputs_embeds = self.prompt_encoder(raw_embeds, position_mask)
            
            # Debug
            if not self._debug_printed:
                print(f"\n[WRAPPER DEBUG]")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Raw embeds shape: {raw_embeds.shape}")
                print(f"  Soft embeds shape: {inputs_embeds.shape}")
                self._debug_printed = True
        
        # 4. Pass to original transformer encoder
        return self.original_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to original encoder."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_encoder, name)


def calculate_metrics(dataset, model, batch_size=8, config=None, save_dir=None):
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
                if label == 'O': continue 

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
    
    print("\n## 📈 Global Metrics (Label2ID Mode, EXCLUDING O)\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

    # Build metrics dict
    metrics = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1
    }

    # Save Results to File
    if save_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Safe extraction of config params for filename
        config = config or {}
        hyper = config.get('hyperparameters', config)
        epochs = hyper.get('num_epochs', 'X')
        lr = hyper.get('learning_rate', 'X')
        prompt_lr = hyper.get('prompt_encoder_lr', 'X')
        
        filename = f"eval_softprompt_v2_ep{epochs}_lr{lr}_plr{prompt_lr}_{timestamp}.md"
        results_dir = Path(save_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
            
        filepath = results_dir / filename
        
        md_content = f"""# Evaluation Report (V2 Soft Prompt)
**Date**: {timestamp}
**Model Type**: Bi-Encoder Soft Prompt V2 (PromptEncoder with Position Mask)

## Configuration
- **Epochs**: {epochs}
- **Learning Rate**: {lr}
- **Prompt Encoder LR**: {prompt_lr}
- **Others LR**: {hyper.get('others_lr', 'N/A')}
- **Weight Decay**: {hyper.get('weight_decay', 'N/A')}
- **Batch Size**: {hyper.get('batch_size', 'N/A')}
- **Backbone**: {config.get('backbone', 'N/A')}

## Metrics
| Metric | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Macro** | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |
| **Micro** | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |

## Dataset Info
- **Test Samples**: {len(dataset)}
- **Labels**: {len(label_list)}
"""
        with open(filepath, "w") as f:
            f.write(md_content)
        
        print(f"\n✅ Results saved to {filepath}")
    
    return metrics


def convert_ids_to_labels(dataset, id_map):
    converted_count = 0
    filtered_count = 0
    
    new_dataset = []
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            label_id = str(label_id)
            if label_id in id_map:
                label_name = id_map[label_id]
                if label_name == "O": 
                    filtered_count += 1
                    continue
                new_ner.append([start, end, label_name])
                converted_count += 1
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset


def find_v2_checkpoint_interactive():
    """Find v2 checkpoint files with interactive selection."""
    savings_dir = Path("savings")
    
    # Look for both new (model_soft_tuned) and old (prompt_encoder) files
    v2_files = list(savings_dir.glob("model_soft_tuned_*.pt")) + list(savings_dir.glob("prompt_encoder_*.pt"))
    
    if not v2_files:
        return None
    
    # Sort by modification time (latest first)
    v2_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("\n" + "="*50)
    print("📁 Available V2 Checkpoints:")
    print("="*50)
    for idx, f in enumerate(v2_files):
        # Try to load metadata from checkpoint itself
        info = ""
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                if 'trainable_params' in ckpt:
                    info += " [FULL MODEL]"
                elif 'state_dict' in ckpt:
                    info += " [PROMPT ONLY]"
                    
                if 'hyperparameters' in ckpt: 
                    hyper = ckpt.get('hyperparameters', {})
                    epochs = hyper.get('num_epochs', '?')
                    lr = hyper.get('learning_rate', '?')
                    prompt_lr = hyper.get('prompt_encoder_lr', '?')
                    info += f" | epochs={epochs}, lr={lr}, prompt_lr={prompt_lr}"
        except:
            pass
        print(f" [{idx}] {f.name}{info}")
    
    print("="*50)
    
    # Interactive selection
    while True:
        try:
            choice = input(f"\n🔢 Select checkpoint [0-{len(v2_files)-1}] (default=0, latest): ").strip()
            if choice == "":
                choice = 0
            else:
                choice = int(choice)
            
            if 0 <= choice < len(v2_files):
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 0 and {len(v2_files)-1}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    checkpoint_path = v2_files[choice]
    print(f"\n✅ Selected checkpoint: {checkpoint_path.name}")
    
    return checkpoint_path


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    if is_running_on_kaggle():
        path = "/kaggle/input/jnlpa15k/"
        model_name = '/kaggle/input/glinerbismall2/'
    else:
        path = "../dataset/"
        model_name = "Ihor/gliner-biomed-bi-small-v1.0"

    test_path = path + "test_dataset_span_bi.json"
    label2id_path = path + "label2id.json"
    
    # CHECKPOINT SELECTION LOGIC (V2 Style) - Interactive
    print("🔍 Searching for V2 checkpoints...")
    checkpoint_path = find_v2_checkpoint_interactive()
    
    if checkpoint_path is None:
        raise FileNotFoundError("No V2 checkpoint found. Looking for files in savings/")
    
    print(f"📁 Checkpoint: {checkpoint_path}")
    
    # Load checkpoint (contains both state_dict and metadata)
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract metadata from checkpoint
    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'trainable_params' in checkpoint):
        # New format: checkpoint contains embedded metadata
        config = {
            'hyperparameters': checkpoint.get('hyperparameters', {}),
            'dataset_info': checkpoint.get('dataset_info', {}),
            'backbone': checkpoint.get('backbone', 'N/A'),
            'architecture': checkpoint.get('architecture', {}),
            'timestamp': checkpoint.get('timestamp', 'N/A'),
            'test_metrics': checkpoint.get('test_metrics', {}),
            'baseline_f1': checkpoint.get('baseline_f1', 'N/A'),
        }
        print(f"\n📋 Loaded config from checkpoint:")
        print(json.dumps(config, indent=2))
    else:
        # Old format: checkpoint is just the state_dict
        config = {}
        print("\n⚠️ Old checkpoint format detected (no embedded metadata)")
    
    # Get label list from config or fallback
    label_list = config.get('dataset_info', {}).get('labels', [])
    num_labels = len(label_list) if label_list else 6  # fallback

    # 1. Load Data
    print("\nLoading datasets and mappings...")
    with open(test_path, "r") as f:
        test_dataset = json.load(f)
    with open(label2id_path, "r") as f:
        label2id = json.load(f)
    
    id2label = {str(v): k for k, v in label2id.items()}
    
    print("\nConverting Test Dataset IDs to Labels...")
    test_dataset = convert_ids_to_labels(test_dataset, id2label)

    # 2. Init Base Model Structure
    print(f"Initializing base architecture {model_name} on {device}...")
    
    try:
        model = GLiNER.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Standard load failed ({e}). Trying fallback with AutoConfig...")
        from transformers import AutoConfig
        config_hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = GLiNER(config_hf).to(device)

    # 3. Get embedding dimensions
    lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    embed_dim = original_embeddings.embedding_dim
    print(f"✅ Label Embedding Dim: {embed_dim}")

    # 4. Create PromptEncoder (V2 style)
    prompt_encoder = PromptEncoder(
        backbone_hidden_size=embed_dim,
        num_labels=num_labels
    ).to(device)
    print(f"✅ PromptEncoder created with dim={embed_dim}, num_labels={num_labels}")
    
    # 5. Inject Wrapper
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)
    model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
    print("✅ SoftPromptLabelEncoderWrapper injected.")

    # 6. Load Weights (Supports both Full Model Tuned and Partial Prompt Only)
    print(f"\nLoading weights...")
    
    if isinstance(checkpoint, dict) and 'trainable_params' in checkpoint:
        # --- CASO 1: Checkpoint Nuovo (Full Tuned: RNN + SpanRep + PromptEncoder) ---
        print("📥 Loading FULL tuned parameters (RNN, SpanRep, PromptEncoder)...")
        # Usiamo strict=False perché nel checkpoint ci sono solo i params allenabili (mancano embeddings congelati)
        trainable_keys = list(checkpoint['trainable_params'].keys())
        print(f"   Total trainable params in checkpoint: {len(trainable_keys)}")
        
        # Mostra quali componenti sono presenti
        components = {'rnn': [], 'span_rep': [], 'prompt_encoder': [], 'other': []}
        for key in trainable_keys:
            if 'rnn' in key.lower():
                components['rnn'].append(key)
            elif 'span_rep' in key.lower():
                components['span_rep'].append(key)
            elif 'prompt_encoder' in key.lower():
                components['prompt_encoder'].append(key)
            else:
                components['other'].append(key)
        
        print(f"   Components found:")
        for comp_name, comp_keys in components.items():
            if comp_keys:
                print(f"     - {comp_name}: {len(comp_keys)} params (e.g., {comp_keys[0]})")
        
        missing, unexpected = model.model.load_state_dict(checkpoint['trainable_params'], strict=False)
        print(f"✅ Weights loaded. Missing (expected frozen): {len(missing)}. Unexpected: {len(unexpected)}")
        
    elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'prompt_encoder_state_dict' in checkpoint):
        # --- CASO 2: Checkpoint Intermedio (Solo Prompt Encoder) ---
        print("⚠️  Loading ONLY Prompt Encoder (Legacy/Partial Checkpoint)...")
        print("    WARNING: RNN and SpanRep weights are NOT loaded - performance will be degraded!")
        
        state_dict_key = 'prompt_encoder_state_dict' if 'prompt_encoder_state_dict' in checkpoint else 'state_dict'
        wrapped_encoder.prompt_encoder.load_state_dict(checkpoint[state_dict_key])
        print(f"✅ Prompt encoder weights loaded from key '{state_dict_key}'.")
        
    else:
        # --- CASO 3: Checkpoint Vecchio (Solo state dict crudo) ---
        print("⚠️  Loading Raw State Dict (Legacy)...")
        print("    WARNING: This appears to be a raw state dict - RNN and SpanRep NOT loaded!")
        wrapped_encoder.prompt_encoder.load_state_dict(checkpoint)
        print("✅ Prompt encoder weights loaded.")

    # 7. Evaluate
    print("\n" + "="*50)
    print("STARTING EVALUATION (V2 Soft Prompt)")
    print("="*50)
    
    model.eval()
    
    # Determine save directory for results (same folder as this script)
    save_dir = Path(__file__).parent
    
    calculate_metrics(
        test_dataset, 
        model, 
        config=config, 
        save_dir=save_dir
    )
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
