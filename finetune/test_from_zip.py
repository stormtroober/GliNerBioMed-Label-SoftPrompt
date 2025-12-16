
import os
import argparse
import zipfile
import tempfile
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from gliner import GLiNER

# ==========================================
# CONFIGURATION & UTILS
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    print(f"{'Label':<30} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Supp.':<8}")
    print("-" * 80)

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
        print(f"{l:<30} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {support[l]:<8}")
        
    macro_p = np.mean(p_s) if p_s else 0.0
    macro_r = np.mean(r_s) if r_s else 0.0
    macro_f1 = np.mean(f1_s) if f1_s else 0.0
    
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values()) 
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print("\n## 📈 Global Metrics (Label2ID Mode)\n")
    print(f"### Performance Summary")
    print(f"| Average Type | Precision | Recall | F1-Score |")
    print(f"|:-------------|----------:|-------:|---------:|")
    print(f"| **Macro**    | {macro_p:.4f} | {macro_r:.4f} | **{macro_f1:.4f}** |")
    print(f"| **Micro**    | {micro_p:.4f} | {micro_r:.4f} | **{micro_f1:.4f}** |")

def convert_ids_to_labels(dataset, id_map):
    converted_count = 0
    filtered_count = 0
    total_spans = 0
    new_dataset = []
    
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            total_spans += 1
            label_id = str(label_id)
            if label_id in id_map:
                label_name = id_map[label_id]
                if label_name == "O": 
                    filtered_count += 1
                    continue
                new_ner.append([start, end, label_name])
                converted_count += 1
            else:
                 print(f"Warning: Label ID not found in map: {label_id}")
    
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Test GLiNER model from a zip file.")
    parser.add_argument("--zip_path", type=str, help="Path to the model zip file.", required=False)
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing model zips (if zip_path not provided).")
    parser.add_argument("--test_data", type=str, default="finetune/jnlpa_test.json", help="Path to test dataset.")
    parser.add_argument("--label2id", type=str, default="label2id.json", help="Path to label2id mapping.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    
    # 1. Resolve Zip Path
    zip_path = args.zip_path
    if not zip_path:
        # Try to find a zip in models_dir
        if os.path.exists(args.models_dir):
            zips = [f for f in os.listdir(args.models_dir) if f.endswith(".zip")]
            if zips:
                zip_path = os.path.join(args.models_dir, zips[0]) # store the first one found
                print(f"No specific zip provided. Found {len(zips)} zips in {args.models_dir}. Using: {zip_path}")
            else:
                 # Check if finetune/models has it
                 alt_models_dir = "finetune/models"
                 if os.path.exists(alt_models_dir):
                     zips = [f for f in os.listdir(alt_models_dir) if f.endswith(".zip")]
                     if zips:
                         zip_path = os.path.join(alt_models_dir, zips[0])
                         print(f"No specific zip provided. Found {len(zips)} zips in {alt_models_dir}. Using: {zip_path}")

    if not zip_path or not os.path.exists(zip_path):
        print(f"Error: Could not find model zip file at '{zip_path}' or in directories.")
        return

    print(f"Testing model from: {zip_path}")

    # 2. Load Data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "r") as f:
        test_dataset = json.load(f)
        
    print(f"Loading label map from {args.label2id}...")
    with open(args.label2id, "r") as f:
        label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    
    # 3. Process Data
    print("Converting Test Dataset IDs to Labels...")
    test_dataset = convert_ids_to_labels(test_dataset, id2label)
    print(f'Final Test dataset size: {len(test_dataset)}')

    # 4. Unzip and Load Model
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting model to temporary directory: {temp_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Check if the extracted content is in a subdir (common issue with zipping)
        # GLiNER expects config.json and model.safetensors/bin in the dir
        model_dir = temp_dir
        # If the zip contains a folder, we might need to go into it
        # Simple heuristic: if no config.json in root, look in subdirs
        if not os.path.exists(os.path.join(model_dir, "config.json")):
             subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
             if len(subdirs) == 1:
                 model_dir = os.path.join(model_dir, subdirs[0])
                 print(f"Found model in subdirectory: {model_dir}")
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        try:
            print(f"Loading GLiNER model from {model_dir}...")
            model = GLiNER.from_pretrained(model_dir).to(device)
            print(f"Model loaded successfully on {device}")
            
            # 5. Evaluate
            calculate_metrics(test_dataset, model, batch_size=args.batch_size)
            
        except Exception as e:
            print(f"Failed to load model or evaluate: {e}")

if __name__ == "__main__":
    main()
