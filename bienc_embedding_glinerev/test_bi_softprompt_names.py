
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from gliner import GLiNER
import torch.nn as nn
from datetime import datetime

# ==========================================================
# 1️⃣ RE-IMPLEMENTATION OF CLASSES (MUST MATCH TRAINING)
# ==========================================================
class PromptPooler(nn.Module):
    def __init__(self, embed_dim, prompt_len, mode="adaptive_avg", max_seq_len=512):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        
        if mode == "adaptive_avg":
            self.pooler = nn.AdaptiveAvgPool1d(prompt_len)
        elif mode == "adaptive_max":
            self.pooler = nn.AdaptiveMaxPool1d(prompt_len)
        elif mode == "attention":
            self.queries = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d":
            self.conv_layers = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d_strided":
            self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    
    def forward(self, x, attention_mask=None):
        B, seq_len, dim = x.shape
        if self.mode in ["adaptive_avg", "adaptive_max"]:
            x_t = x.transpose(1, 2)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).float()
                if self.mode == "adaptive_avg":
                    x_t = x_t * mask_expanded
                else: 
                    x_t = x_t.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = self.pooler(x_t)
            return pooled.transpose(1, 2)
        elif self.mode == "attention":
            queries = self.queries.expand(B, -1, -1)
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)
        elif self.mode == "conv1d":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv_layers(x_t)
            pooled = self.adaptive_pool(conv_out)
            return self.norm(pooled.transpose(1, 2))
        elif self.mode == "conv1d_strided":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv(x_t)
            pooled = self.adaptive_pool(conv_out).transpose(1, 2)
            gate = self.gate(pooled)
            return self.norm(pooled * gate)

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        if hidden_dim is None: hidden_dim = embed_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pooler = None
        if prompt_len is not None:
             self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        if self.pooler is not None:
            x = self.pooler(x, attention_mask)
        return x

class SoftPromptLabelEncoderWrapper(nn.Module):
    def __init__(self, original_encoder, prompt_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.prompt_encoder = prompt_encoder
        self.config = original_encoder.config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        soft_prompts = self.prompt_encoder(input_ids, attention_mask)
        std_embeddings_layer = self.original_encoder.embeddings
        token_embeddings = std_embeddings_layer(input_ids=input_ids)
        cls_embeds = token_embeddings[:, 0:1, :]
        sep_token_id = getattr(self.config, 'sep_token_id', None)
        if sep_token_id is None:
            sep_token_id = 2
        sep_ids = torch.full((input_ids.shape[0], 1), sep_token_id, device=input_ids.device, dtype=torch.long)
        sep_embeds = std_embeddings_layer(input_ids=sep_ids)
        final_embeds = torch.cat([cls_embeds, soft_prompts, sep_embeds], dim=1)
        B = input_ids.shape[0]
        P = soft_prompts.shape[1]
        final_mask = torch.ones((B, 1 + P + 1), device=input_ids.device, dtype=attention_mask.dtype)
        outputs = self.original_encoder(inputs_embeds=final_embeds, attention_mask=final_mask, **kwargs)
        return outputs

    def __getattr__(self, name):
         if name in ["original_encoder", "prompt_encoder", "config", "forward"]:
             return super().__getattr__(name)
         return getattr(self.original_encoder, name)

def patch_bi_encoder_for_soft_prompts(bi_encoder_module):
    import types
    def patched_encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        try:
            prompt_len = self.labels_encoder.model.prompt_encoder.pooler.prompt_len
        except AttributeError:
            prompt_len = 0
        batch_size = input_ids.shape[0]
        if prompt_len > 0:
            total_len = prompt_len + 2
            pooling_mask = torch.ones((batch_size, total_len), device=input_ids.device, dtype=attention_mask.dtype)
        else:
            pooling_mask = attention_mask
        label_kwargs = dict(kwargs)
        label_kwargs.pop("packing_config", None)
        label_kwargs.pop("pair_attention_mask", None)
        label_kwargs["attention_mask"] = attention_mask 
        labels_embeddings = self.labels_encoder(input_ids, *args, **label_kwargs)
        if hasattr(labels_embeddings, "last_hidden_state"):
            labels_embeddings = labels_embeddings.last_hidden_state
        if hasattr(self, "labels_projection"):
            labels_embeddings = self.labels_projection(labels_embeddings)
        labels_embeddings = self.mean_pooling(labels_embeddings, pooling_mask)
        return labels_embeddings
    bi_encoder_module.encode_labels = types.MethodType(patched_encode_labels, bi_encoder_module)
    print("🧩 Monkey-patch applied (Replacement with [CLS]...[SEP])")

# ==========================================================
# 2️⃣ METRICS CALCULATION
# ==========================================================
def calculate_metrics(dataset, model, id2name, batch_size=8):
    # We want to report on Names (e.g. "cell type"), not IDs ("0").
    # Dataset has IDs ("0").
    
    # Identify all entity names present in dataset
    all_label_names = set()
    for d in dataset:
        for x in d['ner']:
            lbl_id = x[2]
            if lbl_id in id2name:
                all_label_names.add(id2name[lbl_id])
    
    if 'O' in all_label_names: all_label_names.remove('O')
    label_name_list = sorted(list(all_label_names))
    
    # Prerequisite: Model needs NAMES to predict (NO Descriptions)
    labels_to_predict = label_name_list
    
    print(f"\nEvaluating on {len(dataset)} samples.")
    print(f"Target Labels (Names): {label_name_list}")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_items = dataset[i:i+batch_size]
        batch_texts = [" ".join(d['tokenized_text']) for d in batch_items]
        
        with torch.no_grad():
            # Pass NAMES directly to model
            batch_preds = [model.predict_entities(t, labels_to_predict, threshold=0.5) for t in batch_texts]

        for idx, item in enumerate(batch_items):
            # Convert GT IDs to Names
            gt_spans = set()
            for s, e, l_id in item['ner']:
                if l_id in id2name:
                    name = id2name[l_id]
                    if name != 'O':
                        gt_spans.add((name, s, e))
            
            for l, _, _ in gt_spans: support[l] += 1
            
            preds = batch_preds[idx]
            tokens = item['tokenized_text']
            char_to_token = {}
            cursor = 0
            for t_i, token in enumerate(tokens):
                end = cursor + len(token)
                for c in range(cursor, end): char_to_token[c] = t_i
                cursor = end + 1
            
            pred_spans = set()
            for p in preds:
                # Prediction label is already a NAME
                pred_name = p['label']
                if pred_name == 'O': continue
                
                if p['start'] in char_to_token and (p['end'] - 1) in char_to_token:
                    pred_spans.add((pred_name, char_to_token[p['start']], char_to_token[p['end'] - 1]))
            
            for l, _, _ in pred_spans & gt_spans: tp[l] += 1
            for l, _, _ in pred_spans - gt_spans: fp[l] += 1
            for l, _, _ in gt_spans - pred_spans: fn[l] += 1
    
    # Report Generation (Using label_name_list)
    p_s, r_s, f1_s = [], [], []
    valid_labels = [l for l in label_name_list if support[l] > 0 or fp[l] > 0]
    report_lines = []
    header = f"{'Label':<30} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Supp.':<8}"
    print("\n" + header)
    print("-" * 80)
    report_lines.append(header)
    report_lines.append("-" * 80)

    for l in valid_labels:
        t, f_p, f_n = tp[l], fp[l], fn[l]
        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_s.append(p); r_s.append(r); f1_s.append(f1)
        line = f"{l:<30} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {support[l]:<8}"
        print(line); report_lines.append(line)
    
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print("\n## 📈 Global Metrics")
    print(f"| Metric | Value |")
    print(f"| Macro F1 | {macro_f1:.4f} |")
    print(f"| Micro F1 | {micro_f1:.4f} |")
    
    return macro_f1, micro_f1, "\n".join(report_lines)

# ==========================================================
# 3️⃣ MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Test Bi-GLiNER with Soft Prompt Adapter (NAMES ONLY)")
    parser.add_argument("--pt_path", type=str, default="savings", help="Path to .pt adapter file or directory")
    parser.add_argument("--dataset_path", type=str, default="../dataset/test_dataset_span_bi.json", help="Path to test dataset")
    parser.add_argument("--label2id", type=str, default="../dataset/label2id.json")
    args = parser.parse_args()

    # Locate PT file
    if os.path.isdir(args.pt_path):
        pt_files = [f for f in os.listdir(args.pt_path) if f.endswith('.pt')]
        if not pt_files:
            print(f"❌ No .pt file found in {args.pt_path}")
            return
        
        # Sort by time (newest first)
        pt_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.pt_path, x)), reverse=True)
        
        print(f"\n📂 Checkpoints found in '{args.pt_path}':")
        for idx, f in enumerate(pt_files):
            print(f"  [{idx}] {f}")
            
        while True:
            try:
                choice = input("\n👉 Select a checkpoint by index: ")
                idx = int(choice)
                if 0 <= idx < len(pt_files):
                    pt_path = os.path.join(args.pt_path, pt_files[idx])
                    break
                else:
                    print("❌ Invalid index. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    else:
        pt_path = args.pt_path

    print(f"📂 Loading Adapter Checkpoint: {pt_path}")
    checkpoint = torch.load(pt_path, map_location=torch.device('cpu')) # Safe load mapping
    
    # Metadata
    meta = checkpoint.get("training_metadata", {})
    arch = checkpoint.get("architecture_params", {})
    base_model_name = meta.get("base_model_name", "Ihor/gliner-biomed-bi-small-v1.0")
    training_mode = meta.get("training_mode", "descriptions")
    
    print(f"🏗️  Base Model: {base_model_name}")
    print(f"⚙️  Architecture: PromptLen={arch.get('prompt_len')}, Pooling={arch.get('pooling_mode')}")
    print(f"ℹ️  Training Mode: {training_mode}")

    if training_mode != "names":
        print("⚠️  WARNING: This script defines evaluation for NAMES, but the checkpoint says it was trained on descriptions (or mode is unspecified). Results may be poor.")
    
    # Load Base Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained(base_model_name).to(device)
    
    # Re-Inject Adapter
    core = model.model
    lbl_enc_model = core.token_rep_layer.labels_encoder.model 
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    
    mlp_prompt_encoder = MLPPromptEncoder(
        original_embeddings=original_embeddings,
        vocab_size=arch['vocab_size'],
        embed_dim=arch['embed_dim'],
        prompt_len=arch['prompt_len'],
        pooling_mode=arch['pooling_mode'],
        dropout=arch['dropout']
    ).to(device)
    
    # Load Adapter Weights
    mlp_prompt_encoder.load_state_dict(checkpoint['prompt_encoder_state_dict'])
    print("✅ Adapter weights loaded successfully.")

    # Load Projection Weights if present
    if "labels_projection_state_dict" in checkpoint:
        if hasattr(core.token_rep_layer, 'labels_projection'):
            core.token_rep_layer.labels_projection.load_state_dict(checkpoint["labels_projection_state_dict"])
            print("✅ Labels Projection weights loaded successfully.")
        else:
            print("⚠️ Checkpoint has projection weights but model has no labels_projection layer.")
    
    # Wrap and Patch
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, mlp_prompt_encoder)
    core.token_rep_layer.labels_encoder.model = wrapped_encoder
    patch_bi_encoder_for_soft_prompts(core.token_rep_layer)
    
    model.eval()

    # Load Data
    print(f"📖 Loading Test Data from {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        full_data = json.load(f)
    
    # Use full test dataset
    test_data = full_data
    print(f"ℹ️  Using full test set (Size: {len(test_data)})")

    # Load Mappings
    print("📖 Loading Mappings/Descriptions...")
    with open(args.label2id) as f: label2id = json.load(f)
    
    # Prepare lookup tables for NAMES
    id2name = {str(v): k for k, v in label2id.items()}
    
    # Evaluate
    macro, micro, report = calculate_metrics(test_data, model, id2name)
    
    # Save Report
    res_dir = "results"
    os.makedirs(res_dir, exist_ok=True)
    out_file = f"{res_dir}/eval_NAMES_{meta.get('timestamp', 'unknown')}.md" # Distinguished filename
    
    with open(out_file, "w") as f:
        f.write(f"# Evaluation Report (NAMES MODE)\n\n")
        f.write(f"**Checkpoint**: `{os.path.basename(pt_path)}`\n")
        f.write(f"**Date**: {datetime.now()}\n\n")
        f.write("## Configuration\n")
        for k, v in meta.items(): f.write(f"- **{k}**: {v}\n")
        f.write("\n## Detailed Metrics\n\n")
        f.write("```\n" + report + "\n```\n")
        f.write(f"\n**Macro F1**: {macro:.4f}\n")
        f.write(f"**Micro F1**: {micro:.4f}\n")
    
    print(f"📝 Report saved to {out_file}")
    
    # Run orderTests.sh
    if os.path.exists("./orderTests.sh"):
        print("📊 Updating Leaderboard (running ./orderTests.sh)...")
        import subprocess
        # Ensure it is executable
        os.chmod("./orderTests.sh", 0o775)
        subprocess.run(["./orderTests.sh"])
    else:
        print("⚠️ orderTests.sh not found in current directory.")

if __name__ == "__main__":
    main()
