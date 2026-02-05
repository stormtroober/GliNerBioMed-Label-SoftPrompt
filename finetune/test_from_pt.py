
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
# CONFIGURATION
# ==========================================
MODELS_DIR = "./models"
DATASET_DIRS = {
    "jnlpba": "../dataset",
    "bc5cdr": "../dataset_bc5cdr"
}

# ==========================================
# METRICS UTIL (Reused) - Without considering 'O' class
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


def list_pt_files(models_dir):
    """List all .pt files in the models directory."""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print(f"Created models directory: {models_dir}")
        return []
    
    pt_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    # Sort by modification time, newest first
    pt_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    return pt_files


def extract_model_metadata(fpath):
    """Extract metadata from a .pt checkpoint file."""
    try:
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)
        metadata = checkpoint.get("training_metadata", {})
        
        # Try to extract encoder type
        encoder_type = metadata.get("encoder_type", None)
        if encoder_type is None:
            # Infer from filename
            fname = os.path.basename(fpath).lower()
            if "bienc" in fname or "_bi_" in fname:
                encoder_type = "bi"
            elif "monoenc" in fname or "_mono_" in fname:
                encoder_type = "mono"
            else:
                encoder_type = "?"
        
        # Try to extract dataset info
        dataset_name = metadata.get("dataset_name", metadata.get("dataset", None))
        dataset_size = metadata.get("dataset_size", metadata.get("train_size", None))
        
        return {
            "encoder_type": encoder_type,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "metadata": metadata
        }
    except Exception as e:
        return {
            "encoder_type": "?",
            "dataset_name": None,
            "dataset_size": None,
            "error": str(e)
        }


def interactive_select_model(models_dir):
    """Interactively select a model from the models directory."""
    pt_files = list_pt_files(models_dir)
    
    if not pt_files:
        print(f"\n❌ No .pt files found in {models_dir}")
        print("Please add your model checkpoints to the 'models' directory.")
        return None
    
    print("\n" + "="*80)
    print("📁 AVAILABLE MODELS")
    print("="*80)
    print("Loading metadata from checkpoints...")
    
    model_info = []
    for idx, fname in enumerate(pt_files, 1):
        fpath = os.path.join(models_dir, fname)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        
        # Extract metadata
        meta = extract_model_metadata(fpath)
        model_info.append((fname, fpath, mod_time, size_mb, meta))
    
    print("-"*80)
    for idx, (fname, fpath, mod_time, size_mb, meta) in enumerate(model_info, 1):
        enc_type = meta.get("encoder_type", "?")
        dataset_name = meta.get("dataset_name")
        dataset_size = meta.get("dataset_size")
        
        # Format encoder display
        enc_display = f"{'bi-enc' if enc_type == 'bi' else 'mono-enc' if enc_type == 'mono' else '?'}"
        
        # Format dataset display
        if dataset_name:
            dataset_display = dataset_name
        elif dataset_size:
            dataset_display = f"size={dataset_size}"
        else:
            dataset_display = "?"
        
        print(f"  [{idx}] {fname}")
        print(f"      📦 {size_mb:.1f} MB | 🔧 Encoder: {enc_display} | 📊 Dataset: {dataset_display}")
        print(f"      📅 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    print("="*80)
    
    while True:
        try:
            choice = input(f"\nSelect model [1-{len(pt_files)}] (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(pt_files):
                selected = pt_files[choice_idx]
                print(f"✅ Selected: {selected}")
                return os.path.join(models_dir, selected)
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(pt_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def interactive_select_encoder():
    """Interactively select encoder type."""
    print("\n" + "="*60)
    print("🔧 ENCODER TYPE")
    print("="*60)
    print("  [1] bi     - Bi-Encoder (gliner-biomed-bi-small-v1.0)")
    print("  [2] mono   - Mono-Encoder (gliner_small-v2.1)")
    print("="*60)
    
    while True:
        choice = input("Select encoder type [1-2]: ").strip()
        if choice == '1':
            return 'bi'
        elif choice == '2':
            return 'mono'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def interactive_select_dataset():
    """Interactively select dataset type."""
    print("\n" + "="*60)
    print("📊 DATASET")
    print("="*60)
    print("  [1] jnlpba  - JNLPBA dataset (../dataset)")
    print("  [2] bc5cdr  - BC5CDR dataset (../dataset_bc5cdr)")
    print("="*60)
    
    while True:
        choice = input("Select dataset [1-2]: ").strip()
        if choice == '1':
            return 'jnlpba'
        elif choice == '2':
            return 'bc5cdr'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    parser = argparse.ArgumentParser(description="Test GLiNER model from a .pt checkpoint.")
    parser.add_argument("--pt_path", type=str, default=None, 
                        help="Path to the model .pt file. If not specified, interactive selection is used.")
    parser.add_argument("--encoder_type", type=str, default=None, choices=["bi", "mono"], 
                        help="Type of encoder: 'bi' for bi-encoder or 'mono' for mono-encoder.")
    parser.add_argument("--dataset", type=str, default=None, choices=["jnlpba", "bc5cdr"],
                        help="Dataset to use: 'jnlpba' or 'bc5cdr'.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, MODELS_DIR)
    
    # List mode
    if args.list:
        pt_files = list_pt_files(models_dir)
        if pt_files:
            print(f"\n📁 Available models in {models_dir}:")
            for f in pt_files:
                print(f"  - {f}")
        else:
            print(f"\n❌ No .pt files found in {models_dir}")
        return
    
    # Interactive or CLI model selection
    if args.pt_path is None:
        pt_path = interactive_select_model(models_dir)
        if pt_path is None:
            print("Exiting.")
            return
    else:
        # If relative path provided, check if it exists in models dir
        if not os.path.isabs(args.pt_path):
            pt_path = os.path.join(models_dir, args.pt_path)
        else:
            pt_path = args.pt_path
        
        if not os.path.exists(pt_path):
            print(f"❌ Error: .pt file not found at {pt_path}")
            return
    
    # Interactive or CLI encoder selection
    if args.encoder_type is None:
        encoder_type = interactive_select_encoder()
    else:
        encoder_type = args.encoder_type
    
    # Interactive or CLI dataset selection
    if args.dataset is None:
        dataset_name = interactive_select_dataset()
    else:
        dataset_name = args.dataset
    
    # Build paths
    dataset_dir = os.path.join(script_dir, DATASET_DIRS[dataset_name])
    
    if encoder_type == "bi":
        test_data_path = os.path.join(dataset_dir, "test_dataset_span_bi.json")
    else:
        test_data_path = os.path.join(dataset_dir, "test_dataset_span_mono.json")
    
    label2id_path = os.path.join(dataset_dir, "label2id.json")
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST CONFIGURATION")
    print("="*60)
    print(f"  Model:        {os.path.basename(pt_path)}")
    print(f"  Encoder:      {encoder_type}")
    print(f"  Dataset:      {dataset_name}")
    print(f"  Test file:    {test_data_path}")
    print(f"  Label2id:     {label2id_path}")
    print(f"  Batch size:   {args.batch_size}")
    print("="*60)
    
    # Confirm
    confirm = input("\nProceed with testing? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("Aborted.")
        return

    if not os.path.exists(pt_path):
        print(f"❌ Error: .pt file not found at {pt_path}")
        return

    print(f"\nLoading checkpoint from: {pt_path}")
    checkpoint = torch.load(pt_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract Metadata
    metadata = checkpoint.get("training_metadata", {})
    print("\n" + "="*50)
    print("TRAINING METADATA")
    print("="*50)
    for k, v in metadata.items():
        print(f"{k}: {v}")
    print("="*50 + "\n")
    
    # Select base model based on encoder type
    if encoder_type == "bi":
        MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
        FALLBACK_MODEL = "Ihor/gliner-biomed-bi-small-v1.0"
    else:  # mono
        MODEL_NAME = "urchade/gliner_small-v2.1"
        FALLBACK_MODEL = "urchade/gliner_small-v2.1"
    
    print(f"Using {encoder_type}-encoder base model: {MODEL_NAME}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        print(f"Initializing base model: {MODEL_NAME}")
        model = GLiNER.from_pretrained(MODEL_NAME)
        
        print("Loading state dict from checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Attempting to load '{FALLBACK_MODEL}' as fallback base...")
        try:
            model = GLiNER.from_pretrained(FALLBACK_MODEL)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
        except Exception as e2:
             print(f"Fallback failed: {e2}")
             return

    # Load Data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, "r") as f:
        test_dataset = json.load(f)
        
    print(f"Loading label map from {label2id_path}...")
    with open(label2id_path, "r") as f:
        label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    
    print("Converting Test Dataset IDs to Labels...")
    test_dataset = convert_ids_to_labels(test_dataset, id2label)
    print(f'Final Test dataset size: {len(test_dataset)}')
    
    macro_f1, micro_f1, report_str = calculate_metrics(test_dataset, model, batch_size=args.batch_size)
    
    # Export Results
    TEST_RESULTS_DIR = os.path.join(script_dir, "test/results")
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(pt_path).replace('.pt', '')
    filename = f"{TEST_RESULTS_DIR}/eval_{model_name}_{dataset_name}_{encoder_type}enc_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        # Test Configuration Section
        f.write("## 🔍 Test Configuration\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| **Model File** | `{os.path.basename(pt_path)}` |\n")
        f.write(f"| **Encoder Type** | `{encoder_type}` |\n")
        f.write(f"| **Dataset** | `{dataset_name}` |\n")
        f.write(f"| **Test File** | `{os.path.basename(test_data_path)}` |\n")
        f.write(f"| **Batch Size** | `{args.batch_size}` |\n")
        f.write(f"| **Base Model** | `{MODEL_NAME}` |\n")
        f.write("\n")
        
        # Training Metadata Section
        f.write("## 🔧 Training Configuration (from checkpoint)\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        for k in sorted(metadata.keys()):
             f.write(f"| **{k}** | `{metadata[k]}` |\n")
        f.write("\n")
        
        # Metrics Section
        f.write("## 📊 Metriche Chiave\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| **Macro F1** | {macro_f1:.4f} |\n")
        f.write(f"| **Micro F1** | {micro_f1:.4f} |\n\n")
        
        # Report Section
        f.write("## 📝 Report\n```\n")
        f.write(report_str)
        f.write("\n```\n")
        
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
