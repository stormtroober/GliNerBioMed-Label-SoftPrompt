
import os
import json
import torch
import torch.nn as nn
from gliner import GLiNER
from gliner.modeling.base import BiEncoderSpanModel
from gliner.modeling.utils import extract_word_embeddings
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Union, List, Dict, Tuple

# ==========================================
# 1. ARCHITETTURA (DA train_bi_softprompt_gliner_v2.py)
# ==========================================

class PromptEncoder(nn.Module):
    def __init__(self, backbone_hidden_size: int, num_labels: int, soft_prompt_length: int = 1, mid_dim: int = None):
        super().__init__()
        self.backbone_hidden_size = backbone_hidden_size
        self.num_labels = num_labels
        self.mlp = nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size)
        self.norm = nn.LayerNorm(self.backbone_hidden_size)
        # Weights are initialized in the standard way, but we will overwrite them
    
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
    def __init__(self, config: Any, from_pretrained: bool = False, cache_dir: Optional[str] = None, 
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
        else:
            # Tentativo 2: Cerca tra i moduli figli un Linear 384->768
            for name, module in self.token_rep_layer.named_children():
                if isinstance(module, nn.Linear):
                    # Verifica le dimensioni
                    if module.in_features == 384 and module.out_features == 768:
                        self.projection_layer = module
                        break

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
        
        # 2e. PROJECTION (384 -> 768)
        if self.projection_layer is not None:
             labels_embeds = self.projection_layer(labels_embeds)

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
        except AttributeError:
            label_embedding_dim = instance.model.token_rep_layer.labels_encoder.config.hidden_size

        prompt_encoder = None
        if prompt_encoder_cls is not None and fixed_labels is not None:
            prompt_encoder = prompt_encoder_cls(backbone_hidden_size=label_embedding_dim, num_labels=len(fixed_labels))

        old_model = instance.model
        instance.model = CustomBiEncoderSpanModel(old_model.config, prompt_encoder=prompt_encoder, fixed_labels=fixed_labels)
        instance.model.load_state_dict(old_model.state_dict(), strict=False)
        instance.model.to(instance.device)
        return instance

# ==========================================
# 2. METRICHE (DA test_bi_softprompt_gliner.py)
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
             batch_preds = model.batch_predict_entities(batch_texts, label_list, threshold=0.5)

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

def convert_ids_to_labels(dataset, id_map):
    converted_count = 0
    new_dataset = []
    for item in dataset:
        new_ner = []
        for start, end, label_id in item['ner']:
            label_id = str(label_id)
            if label_id in id_map:
                label_name = id_map[label_id]
                if label_name == "O": continue
                new_ner.append([start, end, label_name])
                converted_count += 1
        item['ner'] = new_ner
        new_dataset.append(item)
    return new_dataset

# ==========================================
# 3. MAIN
# ==========================================

def main():
    # SETUP PATHS
    base_path = "../dataset/"
    if not os.path.exists(base_path): base_path = "dataset/" # Fallback compatibility
    
    test_path = os.path.join(base_path, "test_dataset_span_bi.json")
    label2id_path = os.path.join(base_path, "label2id.json")
    
    # SPECIFIC ENCODER FILE
    encoder_path = "savings/prompt_encoder_20260204_153524.pt"
    if not os.path.exists(encoder_path):
        # Proviamo path assoluto se stiamo eseguendo dalla root
        encoder_path = "bienc_embedding_glinerev/savings/prompt_encoder_20260204_153524.pt"
    
    print(f"Target Encoder: {encoder_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Cannot find encoder file: {encoder_path}")

    # LOAD DATA
    print("Loading datasets...")
    with open(test_path, "r") as f: test_dataset = json.load(f)
    with open(label2id_path, "r") as f: label2id = json.load(f)
    id2label = {str(v): k for k, v in label2id.items()}
    
    test_dataset = convert_ids_to_labels(test_dataset, id2label)
    
    # DETERMINE ALL LABELS
    all_labels = set()
    for d in test_dataset:
        for _, _, l in d['ner']: all_labels.add(l)
    label_list = sorted(list(all_labels))
    
    # MODEL INIT
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = "Ihor/gliner-biomed-bi-small-v1.0"
    
    print(f"Initializing SoftBiGliner from {model_name}...")
    model = SoftBiGliner.from_pretrained(model_name, prompt_encoder_cls=PromptEncoder, fixed_labels=label_list)
    model.to(device)
    
    # LOAD PROMPT ENCODER WEIGHTS
    print(f"Loading Prompt Encoder weights from {encoder_path}...")
    state_dict = torch.load(encoder_path, map_location=device)
    
    # Verify keys match
    model_keys = set(model.model.prompt_encoder.state_dict().keys())
    load_keys = set(state_dict.keys())
    
    if model_keys != load_keys:
        print(f"⚠️ Warning: Keys mismatch. Model: {len(model_keys)}, File: {len(load_keys)}")
        print(f"Missing in file: {model_keys - load_keys}")
        print(f"Extra in file: {load_keys - model_keys}")
    
    model.model.prompt_encoder.load_state_dict(state_dict)
    print("✅ Weights loaded successfully.")
    
    # EVALUATE
    print("\n" + "="*50)
    print("STARTING EVALUATION")
    print("="*50)
    
    model.eval()
    calculate_metrics(test_dataset, model)

if __name__ == "__main__":
    main()
