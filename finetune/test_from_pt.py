
import os
import datetime
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from gliner import GLiNER

# ==========================================
# METRICS UTIL (Reused)
# ==========================================
def calculate_metrics(dataset, model, batch_size=8):
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    all_labels = set()
    for d in dataset:
        for x in d['ner']: 
            all_labels.add(x[2])
    
    if 'O' in all_labels:
        all_labels.remove('O')
        
    label_list = sorted(list(all_labels))
    print(f"\nEvaluating on {len(dataset)} samples with {len(label_list)} labels: {label_list}")
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    
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

    p_s, r_s, f1_s = [], [], []
    valid_labels = [l for l in label_list if support[l] > 0 or fp[l] > 0] 
    
    print("\n### Per-Label Performance")
    header = f"{'Label':<30} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Supp.':<8}"
    print(header)
    print("-" * 80)
    
    report_lines = []
    report_lines.append(header)
    report_lines.append("-" * 80)

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
        line = f"{l:<30} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {support[l]:<8}"
        print(line)
        report_lines.append(line)
        
    macro_p = np.mean(p_s) if p_s else 0.0
    macro_r = np.mean(r_s) if r_s else 0.0
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values()) 
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print("\n## 📈 Global Metrics (From .pt checkpoint)\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

    return macro_f1, micro_f1, "\n".join(report_lines)

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
            else:
                 pass # Warning could be logged
    
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Test GLiNER model from a .pt checkpoint.")
    parser.add_argument("--pt_path", type=str, default="models/", help="Path to the model .pt file.")
    parser.add_argument("--test_data", type=str, default="../dataset/test_dataset_span_mono.json", help="Path to test dataset.")
    parser.add_argument("--label2id", type=str, default="../dataset/label2id.json", help="Path to label2id mapping.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    
    # Handle directory input for pt_path
    if os.path.isdir(args.pt_path):
        print(f"Searching for .pt files in {args.pt_path}...")
        pt_files = [f for f in os.listdir(args.pt_path) if f.endswith('.pt')]
        if not pt_files:
            print(f"Error: No .pt file found in {args.pt_path}")
            return
        # Sort by modification time, newest first
        pt_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.pt_path, x)), reverse=True)
        
        print(f"Found {len(pt_files)} checkpoints. Using the latest one:")
        for idx, f in enumerate(pt_files[:3]):
            print(f"  - {f}")
            
        args.pt_path = os.path.join(args.pt_path, pt_files[0])
        print(f"Selected checkpoint: {args.pt_path}")

    if not os.path.exists(args.pt_path):
        print(f"Error: .pt file not found at {args.pt_path}")
        return

    print(f"Loading checkpoint from: {args.pt_path}")
    checkpoint = torch.load(args.pt_path, map_location=torch.device('cpu'), weights_only=False) # Load keys first
    
    # Extract Metadata
    metadata = checkpoint.get("training_metadata", {})
    print("\n" + "="*50)
    print("TRAINING METADATA")
    print("="*50)
    for k, v in metadata.items():
        print(f"{k}: {v}")
    print("="*50 + "\n")
    
    MODEL_NAME = "urchade/gliner_small-v2.1"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        print(f"Initializing base model: {MODEL_NAME}")
        # Initialize model structure
        model = GLiNER.from_pretrained(MODEL_NAME)
        
        # Load State Dict
        print("Loading state dict from checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load 'urchade/gliner_small-v2.1' as fallback base...")
        try:
            model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
        except Exception as e2:
             print(f"Fallback failed: {e2}")
             return

    # Load Data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "r") as f:
        test_dataset = json.load(f)
        
    print(f"Loading label map from {args.label2id}...")
    with open(args.label2id, "r") as f:
        label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    
    print("Converting Test Dataset IDs to Labels...")
    test_dataset = convert_ids_to_labels(test_dataset, id2label)
    print(f'Final Test dataset size: {len(test_dataset)}')
    
    macro_f1, micro_f1, report_str = calculate_metrics(test_dataset, model, batch_size=args.batch_size)
    
    # Export Results
    TEST_RESULTS_DIR = "test/results"
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{TEST_RESULTS_DIR}/eval_from_pt_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        # Config Section
        f.write("## 🔧 Training Configuration\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        # Sort keys for consistency
        for k in sorted(metadata.keys()):
             f.write(f"| **{k}** | `{metadata[k]}` |\n")
        f.write("\n")
        
        # Metrics Section
        f.write("## Metriche Chiave\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| **Macro F1** | {macro_f1:.4f} |\n")
        f.write(f"| **Micro F1** | {micro_f1:.4f} |\n\n")
        
        # Report Section
        f.write("## Report\n```\n")
        f.write(report_str)
        f.write("\n```\n")
        
    print(f"💾 Saved to {filename}")

if __name__ == "__main__":
    main()
