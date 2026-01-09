
import os
import json
import torch
import torch.nn as nn
from gliner import GLiNER
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# ==========================================
# UTIL & CLASSES
# ==========================================

def is_running_on_kaggle():
    return os.path.exists('/kaggle/input')

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, hidden_dim=None, dropout=0.4):
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

    def forward(self, input_ids):
        # 1. Get original embeddings
        x = self.embedding(input_ids)
        
        # 2. Compute Transform
        # Residual connection: x + mlp(x)
        delta = self.mlp(x)
        
        # 3. Apply Mask: Do NOT transform [CLS](101), [SEP](102), [PAD](0)
        mask = (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
        mask = mask.unsqueeze(-1).expand_as(delta).float()
        delta = delta * mask
        
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

def calculate_metrics(dataset, model, batch_size=8, config=None):
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

    # Save Results to File
    if config:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Safe extraction of config params for filename
        epochs = config.get('num_train_epochs', 'X')
        lr = config.get('learning_rate', 'X')
        alpha = config.get('focal_loss_alpha', 'X')
        gamma = config.get('focal_loss_gamma', 'X')
        
        filename = f"eval_softprompt_ep{epochs}_lr{lr}_a{alpha}_g{gamma}_{timestamp}.md"
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        filepath = os.path.join(results_dir, filename)
        
        md_content = f"""# Evaluation Report
**Date**: {timestamp}
**Model Type**: Bi-Encoder Soft Prompt

## Configuration
- **Epochs**: {epochs}
- **Learning Rate**: {lr}
- **Focal Alpha**: {alpha}
- **Focal Gamma**: {gamma}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}
- **Batch Size**: {config.get('per_device_train_batch_size', 'N/A')}

## Metrics
| Metric | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Macro** | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |
| **Micro** | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |

## Detailed Metrics by Label
The detailed per-label metrics are available in the console logs.
"""
        with open(filepath, "w") as f:
            f.write(md_content)
        
        print(f"\n✅ Results saved to {filepath}")


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
    
    # CHECKPOINT SELECTION LOGIC
    savings_dir = "savings"
    print(f"Checking for checkpoints in: {os.path.abspath(savings_dir)}")
    
    if not os.path.exists(savings_dir):
        # Fallback for compatibility or error
        raise FileNotFoundError(f"Savings directory not found at {savings_dir}")
        
    checkpoints = [f for f in os.listdir(savings_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoint files found in {savings_dir}")
        
    # Sort by modification time (latest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(savings_dir, x)), reverse=True)
    
    print("\nAvailable Checkpoints:")
    for idx, cp in enumerate(checkpoints):
        print(f" [{idx}] {cp}")
        
    selected_checkpoint = checkpoints[0]
    checkpoint_path = os.path.join(savings_dir, selected_checkpoint)
    print(f"\n✅ Auto-selected latest checkpoint: {selected_checkpoint}")

    # 1. Load Data
    print("Loading datasets and mappings...")
    with open(test_path, "r") as f:
        test_dataset = json.load(f)
    with open(label2id_path, "r") as f:
        label2id = json.load(f)
    
    id2label = {str(v): k for k, v in label2id.items()}
    
    print("\nConverting Test Dataset IDs to Labels...")
    test_dataset = convert_ids_to_labels(test_dataset, id2label)

    # 2. Init Base Model Structure
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Initializing base architecture {model_name} on {device}...")
    # We use from_pretrained directly as it works in training. 
    # Warning: This might print "Missing keys" / "Unexpected keys" if the HF hub weights 
    # don't perfectly match the GLiNER class structure. 
    # IGNORE these warnings, because we will immediately overwrite the weights with our checkpoint.
    try:
        model = GLiNER.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Standard load failed ({e}). Trying fallback with AutoConfig...")
        from transformers import AutoConfig
        # Fallback: Try loading config via AutoConfig if GLiNERConfig fails serialization
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = GLiNER(config).to(device)

    # 3. Load Checkpoint (Full Model)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print("Checkpoint Config:", config)

    # 4. Reconstruct Components
    lbl_enc_model = model.model.token_rep_layer.labels_encoder.model
    # We need to access the original embeddings from the initialized (random) model 
    # just to get sizes, or use config. 
    # NOTE: The weights will be overwritten by load_state_dict, so random init is fine.
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    
    prompt_encoder = MLPPromptEncoder(
        original_embeddings, 
        config['vocab_size'], 
        config['embed_dim'], 
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # 5. Inject Wrapper
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_encoder)
    model.model.token_rep_layer.labels_encoder.model = wrapped_encoder
    print("✅ SoftPromptLabelEncoderWrapper injected.")

    # 6. Load ALL Weights
    # This loads BERT, PromptEncoder, Projection, etc.
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Full model weights loaded.")

    # 7. Evaluate
    print("\n" + "="*50)
    print("STARTING EVALUATION")
    print("="*50)
    
    model.eval()
    calculate_metrics(test_dataset, model, config=config)

if __name__ == "__main__":
    main()
