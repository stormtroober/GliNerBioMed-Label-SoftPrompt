# -*- coding: utf-8 -*-
"""
TEST Mono-Encoder MLP Prompt Encoder (NO 'O' CLASS IN METRICS).
Loads the best saved model and evaluates it on the token-level test dataset.
Excludes 'O' class from F1 calculations.
Includes span-based evaluation.
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from torch import nn
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from collections import Counter, defaultdict
import numpy as np
import os
from tqdm import tqdm
import subprocess
import datetime
import time

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#DATASET = '../dataset_bc5cdr/'
DATASET = '../dataset/'


TEST_PATH = DATASET + 'test_dataset_tknlvl_mono.json'
TEST_SPAN_PATH = DATASET + 'test_dataset_span_mono.json'
LABEL2DESC_PATH = DATASET + 'label2desc.json'
LABEL2ID_PATH = DATASET + 'label2id.json'
MODEL_NAME = "urchade/gliner_small-v2.1"
SAVINGS_DIR = "savings"
TEST_RESULTS_DIR = "test_results"
SHELL_SCRIPT_NAME = "../embedding_injection/orderTests.sh"

# ==========================================================
# 1️⃣ CLASSE MLP (Identica per caricare pesi)
# ==========================================================
class PromptPooler(nn.Module):
    """Riduce la sequenza da (B, seq_len, dim) a (B, prompt_len, dim)"""
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


class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg",
                 max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
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
            self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode, max_seq_len=max_seq_len)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        if self.pooler is not None:
            x = self.pooler(x, attention_mask)
        return x

# ==========================================================
# 2️⃣ SELEZIONE CHECKPOINT
# ==========================================================
def select_checkpoint_interactive():
    if not os.path.exists(SAVINGS_DIR): return None
    ckpts = [f for f in os.listdir(SAVINGS_DIR) if f.endswith('.pt')]
    if not ckpts: return None
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(SAVINGS_DIR, x)), reverse=True)
    
    print("\n📦 CHECKPOINT DISPONIBILI:")
    for i, c in enumerate(ckpts[:10]):
        ckpt_path = os.path.join(SAVINGS_DIR, c)
        try:
            # Carica solo la configurazione su CPU
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            conf = checkpoint.get('config', {})
            input_dir = conf.get('input_dir', 'sconosciuto')
            # Pulisce un po' il path se è lungo (es. Kaggle)
            if input_dir.startswith('/kaggle/input/'):
                input_dir = input_dir.replace('/kaggle/input/', '')
            elif input_dir == '../':
                # dataset di default per i run locali
                input_dir = 'dataset locale'
            print(f"{i+1}. {c} | Dataset: {input_dir}")
        except Exception as e:
            print(f"{i+1}. {c} | Dataset: Errore lettura")
            
    try:
        sel = int(input("\n👉 Scegli numero [1]: ") or 1) - 1
        return os.path.join(SAVINGS_DIR, ckpts[sel])
    except: return os.path.join(SAVINGS_DIR, ckpts[0])

ckpt_path = select_checkpoint_interactive()
if not ckpt_path:
    print("❌ Nessun checkpoint trovato.")
    exit()

print(f"Loading: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=DEVICE)
train_config = checkpoint.get('config', {})

PROMPT_LEN = train_config.get('prompt_len', 16)
POOLING_MODE = train_config.get('pooling_mode', 'adaptive_avg')
MAX_TEXT_LEN = train_config.get('max_text_len', 384)

print(f"✅ Config: PLen={PROMPT_LEN}, Mode={POOLING_MODE}, MaxTxt={MAX_TEXT_LEN}")

# ==========================================================
# 3️⃣ LOAD MODEL
# ==========================================================
print("📦 Loading Model and Wrapper...")
model_wrapper = GLiNER.from_pretrained(MODEL_NAME)
model = model_wrapper.model
tokenizer = model_wrapper.data_processor.transformer_tokenizer

backbone = model.token_rep_layer.bert_layer.model
for p in backbone.parameters(): p.requires_grad = False
backbone.to(DEVICE).eval()

original_word_embeddings = backbone.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

prompt_encoder = MLPPromptEncoder(
    original_word_embeddings, 
    vocab_size, 
    embed_dim, 
    prompt_len=PROMPT_LEN, 
    pooling_mode=POOLING_MODE
).to(DEVICE)

prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
prompt_encoder.eval()
print("✨ Model Loaded.")

# ==========================================================
# 4️⃣ PREPARE PROMPTS
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_names = [id2label[i] for i in range(len(label2id))]
num_labels = len(label2id)

desc_texts = [label2desc[name] for name in label_names]
desc_inputs = tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True)
desc_input_ids = desc_inputs["input_ids"].to(DEVICE)
desc_attn_mask = desc_inputs["attention_mask"].to(DEVICE)

with torch.no_grad():
    soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask)
    soft_prompts_flat = soft_prompts.view(-1, embed_dim)
    prompts_len_total = soft_prompts_flat.shape[0]

# ==========================================================
# 5️⃣ SPAN EVALUATION UTILITIES
# ==========================================================
def get_word_boundary_flags(subword_tokens):
    """
    Restituisce True per il primo subword di ogni parola word-level,
    False per token speciali e continuazioni.
    """
    flags = []
    for tok in subword_tokens:
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            flags.append(False)
        elif tok.startswith("▁") or tok.startswith("Ġ"):
            flags.append(True)
        elif not flags:
            flags.append(True)
        else:
            flags.append(False)
    return flags


def subword_preds_to_word_preds(subword_tokens, subword_preds, subword_labels):
    """
    Mappa predizioni subword → word-level usando strategia 'first-subword'.
    """
    flags = get_word_boundary_flags(subword_tokens)
    word_preds, word_trues = [], []
    for flag, pred, true in zip(flags, subword_preds, subword_labels):
        if flag and true != -100:
            word_preds.append(pred)
            word_trues.append(true)
    return word_preds, word_trues


def extract_spans_from_word_seq(word_labels, background_label_id):
    """
    Estrae span contigui dello stesso label (diverso da background)
    come set di tuple (start, end, label_id). Indici 0-based inclusivi.
    """
    spans = set()
    n = len(word_labels)
    i = 0
    while i < n:
        lbl = word_labels[i]
        if lbl != background_label_id:
            j = i
            while j < n and word_labels[j] == lbl:
                j += 1
            spans.add((i, j - 1, lbl))
            i = j
        else:
            i += 1
    return spans


def compute_span_metrics(tp_dict, fp_dict, fn_dict, support_dict, id2label, background_label_id):
    """Calcola metriche span-based per label e aggregate (macro/micro)."""
    all_ids = sorted(set(list(tp_dict.keys()) + list(fp_dict.keys()) + list(fn_dict.keys())))
    all_ids = [lid for lid in all_ids if lid != background_label_id]

    p_list, r_list, f1_list = [], [], []
    report_lines = []
    header = f"{'Label':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>10}"
    report_lines.append(header)
    report_lines.append("-" * 65)

    for lid in all_ids:
        tp = tp_dict[lid]; fp = fp_dict[lid]; fn = fn_dict[lid]; sup = support_dict[lid]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        name = id2label.get(lid, str(lid))
        p_list.append(p); r_list.append(r); f1_list.append(f1)
        report_lines.append(f"{name:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {sup:>10}")

    macro_p  = float(np.mean(p_list))  if p_list  else 0.0
    macro_r  = float(np.mean(r_list))  if r_list  else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    total_tp = sum(tp_dict[l] for l in all_ids)
    total_fp = sum(fp_dict[l] for l in all_ids)
    total_fn = sum(fn_dict[l] for l in all_ids)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, report_lines

# ==========================================================
# 6️⃣ CARICAMENTO TEST SET
# ==========================================================
with open(TEST_PATH) as f: test_data = json.load(f)
print(f"\n📚 Caricati {len(test_data)} record token-level")

span_data = None
if os.path.exists(TEST_SPAN_PATH):
    with open(TEST_SPAN_PATH) as f: span_data = json.load(f)
    print(f"📚 Caricati {len(span_data)} record span-level")
    if len(test_data) != len(span_data):
        print(f"⚠️  Numero esempi diverso ({len(test_data)} vs {len(span_data)}). "
              "La valutazione span potrebbe essere inaccurata.")
else:
    print(f"⚠️  File span non trovato in {TEST_SPAN_PATH}. Valutazione span saltata.")

# ==========================================================
# 7️⃣ TEST LOOP
# ==========================================================
print(f"\n🔍 Valutazione su {len(test_data)} record...")

y_true = []
y_pred = []

# Identify O class
ignore_index = -1
for idx, name in enumerate(label_names):
    if name == 'O': 
        ignore_index = idx
        break

all_label_ids = list(range(len(label_names)))
if ignore_index != -1:
    relevant_label_ids = [i for i in all_label_ids if i != ignore_index]
    relevant_label_names = [label_names[i] for i in relevant_label_ids]
    print(f"ℹ️  Esclusione classe 'O' (ID: {ignore_index}) dalle metriche token-level.")
else:
    relevant_label_ids = all_label_ids
    relevant_label_names = label_names

# Span accumulatori
background_id = label2id.get("O", ignore_index if ignore_index != -1 else 5)
span_tp = defaultdict(int)
span_fp = defaultdict(int)
span_fn = defaultdict(int)
span_support = defaultdict(int)
n_span_skipped = 0

total_records = len(test_data)
checkpoint_interval = max(1, total_records // 5)
print(f"\n📊 Mostro metriche ogni {checkpoint_interval} record (~20%)\n")

print(f"Testing...")
infer_start_time = time.time()
with torch.no_grad():
    for idx, rec in enumerate(tqdm(test_data), 1):
        tokens = rec["tokens"]
        labels = rec["labels"]
        
        inp_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Rimuovi token speciali (CLS/SEP) già presenti nel dataset
        if len(inp_ids) > 0 and inp_ids[0] == tokenizer.cls_token_id:
            inp_ids = inp_ids[1:]
            labels = labels[1:]
        
        if len(inp_ids) > 0 and inp_ids[-1] == tokenizer.sep_token_id:
            inp_ids = inp_ids[:-1]
            labels = labels[:-1]
        
        # Tronca al max text len
        if len(inp_ids) > MAX_TEXT_LEN:
            inp_ids = inp_ids[:MAX_TEXT_LEN]
            labels = labels[:MAX_TEXT_LEN]
        
        if len(inp_ids) == 0:
            n_span_skipped += 1
            continue
            
        input_tensor = torch.tensor([inp_ids], device=DEVICE)
        attn_mask = torch.ones_like(input_tensor)
        
        # --- FORWARD PASS ---
        batch_soft_prompts = soft_prompts_flat.unsqueeze(0)
        text_embeds = backbone.embeddings(input_tensor)
        
        cls_token = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
        sep_token = torch.tensor([[tokenizer.sep_token_id]], device=DEVICE)
        cls_embed = backbone.embeddings(cls_token)
        sep_embed = backbone.embeddings(sep_token)
        
        inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
        
        B = 1
        prompt_mask = torch.ones((B, prompts_len_total), device=DEVICE)
        cls_mask = torch.ones((B, 1), device=DEVICE)
        sep_mask = torch.ones((B, 1), device=DEVICE)
        full_mask = torch.cat([cls_mask, prompt_mask, sep_mask, attn_mask, sep_mask], dim=1)
        
        outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2))
        sequence_output = outputs.last_hidden_state
        
        # Estrai rappresentazioni testo
        text_start = 1 + prompts_len_total + 1
        text_end = text_start + len(inp_ids)
        text_reps = sequence_output[:, text_start:text_end, :]
        
        # Estrai rappresentazioni prompt
        prompt_reps_seq = sequence_output[:, 1:1+prompts_len_total, :]
        prompt_reps_reshaped = prompt_reps_seq.view(1, num_labels, PROMPT_LEN, embed_dim)
        prompt_vectors = prompt_reps_reshaped.mean(dim=2)
        
        H_text = F.normalize(text_reps, dim=-1)
        H_prompts = F.normalize(prompt_vectors, dim=-1)
        logits = torch.bmm(H_text, H_prompts.transpose(1, 2))
        
        preds = logits.argmax(-1).squeeze(0).cpu().tolist()
        
        # Token-level raccolta
        for p, t in zip(preds, labels):
            if t != -100:
                y_true.append(t)
                y_pred.append(p)

        # ---- Span-based evaluation ----
        # Nota: tokens qui sono già stati privati di CLS/SEP iniziale,
        # quindi usiamo i tokens troncati corrispondenti
        rec_tokens_trimmed = tokens
        # Rimuovi CLS/SEP anche dai tokens stringa per il boundary detection
        if rec_tokens_trimmed and rec_tokens_trimmed[0] in ("[CLS]", "<s>"):
            rec_tokens_trimmed = rec_tokens_trimmed[1:]
        if rec_tokens_trimmed and rec_tokens_trimmed[-1] in ("[SEP]", "</s>"):
            rec_tokens_trimmed = rec_tokens_trimmed[:-1]
        rec_tokens_trimmed = rec_tokens_trimmed[:MAX_TEXT_LEN]

        if span_data is not None:
            rec_idx = idx - 1  # idx parte da 1 nel loop
            if rec_idx < len(span_data):
                span_rec = span_data[rec_idx]
                gt_ner = span_rec.get("ner", [])

                gt_spans = set()
                for s, e, lbl_str in gt_ner:
                    lbl_int = int(lbl_str)
                    if lbl_int != background_id:
                        gt_spans.add((s, e, lbl_int))
                        span_support[lbl_int] += 1

                word_preds, _ = subword_preds_to_word_preds(rec_tokens_trimmed, preds, labels)
                pred_spans = extract_spans_from_word_seq(word_preds, background_id)

                for span in pred_spans:
                    if span in gt_spans:
                        span_tp[span[2]] += 1
                    else:
                        span_fp[span[2]] += 1
                for span in gt_spans:
                    if span not in pred_spans:
                        span_fn[span[2]] += 1
            else:
                n_span_skipped += 1

        # Progress checkpoint
        if idx % checkpoint_interval == 0 or idx == total_records:
            if len(y_true) > 0:
                progress = (idx / total_records) * 100
                current_macro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
                )[2]
                current_micro_f1 = precision_recall_fscore_support(
                    y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
                )[2]
                print(f"\n [{progress:5.1f}%] Macro F1: {current_macro_f1:.4f} | Micro F1: {current_micro_f1:.4f} | Tokens: {len(y_true):,}")

if n_span_skipped > 0:
    print(f"⚠️  {n_span_skipped} record saltati nella valutazione span.")

infer_time = time.time() - infer_start_time
samples_per_sec = len(test_data) / infer_time if infer_time > 0 else 0
tokens_per_sec = len(y_true) / infer_time if infer_time > 0 else 0

# ==========================================================
# 8️⃣ METRICS & SAVING
# ==========================================================
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
)
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
)

print(f"\n🏆 GLOBAL TOKEN-LEVEL METRICS (No O Class):")
print(f"   • Tempo Inferenza: {infer_time:.2f} s")
print(f"   • Velocità: {samples_per_sec:.2f} samples/s | {tokens_per_sec:.2f} tokens/s")
print(f"   • MACRO: F1={macro_f1:.4f}")
print(f"   • MICRO: F1={micro_f1:.4f}")

class_report = classification_report(
    y_true, y_pred, target_names=relevant_label_names, labels=relevant_label_ids, zero_division=0
)
print(class_report)

# Span metrics
span_macro_p = span_macro_r = span_macro_f1 = 0.0
span_micro_p = span_micro_r = span_micro_f1 = 0.0
span_report_lines = []

if span_data is not None:
    span_macro_p, span_macro_r, span_macro_f1, \
    span_micro_p, span_micro_r, span_micro_f1, \
    span_report_lines = compute_span_metrics(
        span_tp, span_fp, span_fn, span_support, id2label, background_id
    )
    print(f"\n{'='*60}")
    print(f"🎯 METRICHE SPAN-BASED (Exact Match, escluso 'O')")
    print(f"{'='*60}")
    print(f"  Macro P: {span_macro_p:.4f} | Macro R: {span_macro_r:.4f} | Macro F1: {span_macro_f1:.4f}")
    print(f"  Micro P: {span_micro_p:.4f} | Micro R: {span_micro_r:.4f} | Micro F1: {span_micro_f1:.4f}")
    print(f"\n📋 REPORT SPAN PER CLASSE:")
    for line in span_report_lines:
        print(line)

# Export
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{TEST_RESULTS_DIR}/eval_mono_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    config = checkpoint.get('config', {})
    
    config_table = "| Parameter | Value |\n|---|---|\n"
    for k in sorted(config.keys()):
        config_table += f"| **{k}** | `{config[k]}` |\n"

    f.write(f"## 🔧 Training Configuration\n")
    f.write(f"{config_table}\n\n")

    f.write(f"## Metriche Chiave\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| **Tempo Inferenza** | {infer_time:.2f} s |\n")
    f.write(f"| **Samples/s** | {samples_per_sec:.2f} |\n")
    f.write(f"| **Tokens/s** | {tokens_per_sec:.2f} |\n")
    f.write(f"| **Macro F1** | {macro_f1:.4f} |\n")
    f.write(f"| **Micro F1** | {micro_f1:.4f} |\n\n")
    
    f.write(f"## Report Token-Level\n```\n{class_report}\n```\n")

    if span_data is not None:
        f.write(f"\n## Metriche Span-Based (Exact Match, escluso 'O')\n")
        f.write(f"| Metric | Precision | Recall | F1 |\n")
        f.write(f"|--------|-----------|--------|---------|\n")
        f.write(f"| **Macro** | {span_macro_p:.4f} | {span_macro_r:.4f} | **{span_macro_f1:.4f}** |\n")
        f.write(f"| **Micro** | {span_micro_p:.4f} | {span_micro_r:.4f} | **{span_micro_f1:.4f}** |\n")
        f.write(f"\n### Report Span per Classe\n```\n")
        for line in span_report_lines:
            f.write(line + "\n")
        f.write("```\n")

print(f"💾 Saved to {filename}")

# Run Order Script
print(f"\n🚀 Running Order Script...")
if os.path.exists(SHELL_SCRIPT_NAME):
    try:
        subprocess.run(["bash", SHELL_SCRIPT_NAME], check=True)
        print("✅ Sorted.")
    except Exception as e:
        print(f"❌ Error running script: {e}")
else:
    if os.path.exists("orderTests.sh"):
         subprocess.run(["bash", "orderTests.sh"], check=True)
    elif os.path.exists("../embedding_injection/orderTests.sh"):
         subprocess.run(["bash", "../embedding_injection/orderTests.sh"], check=True)
    else:
        print("⚠️ Order script not found.")