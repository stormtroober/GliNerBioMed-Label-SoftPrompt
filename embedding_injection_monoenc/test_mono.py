# -*- coding: utf-8 -*-
"""
TEST Mono-Encoder MLP Prompt Encoder (NO 'O' CLASS IN METRICS).
Loads the best saved model and evaluates it on the token-level test dataset.
Excludes 'O' class from F1 calculations.
"""

import json
import torch
import torch.nn.functional as F
from gliner import GLiNER
from torch import nn
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from collections import Counter
import os
from tqdm import tqdm
import subprocess
import datetime

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "../dataset/test_dataset_tokenlevel.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "urchade/gliner_small-v2.1"
SAVINGS_DIR = "savings"
TEST_RESULTS_DIR = "test_results"
SHELL_SCRIPT_NAME = "../embedding_injection/orderTests.sh" # Assumi che siamo in gliner2

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
    for i, c in enumerate(ckpts[:10]): print(f"{i+1}. {c}")
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

PROMPT_LEN = train_config.get('prompt_len', 16) # Default from train_mono if missing
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
# Freeze backbone
for p in backbone.parameters(): p.requires_grad = False
backbone.to(DEVICE).eval()

original_word_embeddings = backbone.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

# Load Prompt Encoder
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

# Pre-calculate Prompt Embeddings (Vector per Label)
with torch.no_grad():
    soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask) # (NumLabels, PLen, D)
    soft_prompts_flat = soft_prompts.view(-1, embed_dim)
    prompts_len_total = soft_prompts_flat.shape[0]

# ==========================================================
# 5️⃣ TEST LOOP
# ==========================================================
with open(TEST_PATH) as f: test_data = json.load(f)
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
    print(f"ℹ️  Esclusione classe 'O' (ID: {ignore_index}) dalle metriche.")
else:
    relevant_label_ids = all_label_ids
    relevant_label_names = label_names

total_records = len(test_data)
checkpoint_interval = max(1, total_records // 5)

print(f"\n📊 Mostro metriche ogni {checkpoint_interval} record (~20%)\n")

print(f"Testing...")
with torch.no_grad():
    for idx, rec in enumerate(tqdm(test_data), 1):
        tokens = rec["tokens"]
        labels = rec["labels"]
        
        # Truncate to Max Text Len
        if len(tokens) > MAX_TEXT_LEN:
            tokens = tokens[:MAX_TEXT_LEN]
            labels = labels[:MAX_TEXT_LEN]
            
        inp_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([inp_ids], device=DEVICE) # (1, SeqLen)
        attn_mask = torch.ones_like(input_tensor) # (1, SeqLen)
        
        # --- FORWARD PASS (Same as Train) ---
        # 1. Expand Prompts for Batch (Batch=1)
        batch_soft_prompts = soft_prompts_flat.unsqueeze(0) # (1, TotalPLen, D)
        
        # 2. Text Embeddings
        text_embeds = backbone.embeddings(input_tensor) # (1, TextLen, D)
        
        # 3. Conccat: [CLS] Prompts [SEP] Text [SEP]
        cls_token = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
        sep_token = torch.tensor([[tokenizer.sep_token_id]], device=DEVICE)
        cls_embed = backbone.embeddings(cls_token)
        sep_embed = backbone.embeddings(sep_token)
        
        inputs_embeds = torch.cat([cls_embed, batch_soft_prompts, sep_embed, text_embeds, sep_embed], dim=1)
        
        # Mask
        B = 1
        prompt_mask = torch.ones((B, prompts_len_total), device=DEVICE)
        cls_mask = torch.ones((B, 1), device=DEVICE)
        sep_mask = torch.ones((B, 1), device=DEVICE)
        full_mask = torch.cat([cls_mask, prompt_mask, sep_mask, attn_mask, sep_mask], dim=1)
        
        # Encoder
        outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask.unsqueeze(1).unsqueeze(2))
        sequence_output = outputs.last_hidden_state
        
        # Extract Text Reps
        text_start = 1 + prompts_len_total + 1
        text_end = text_start + len(inp_ids)
        text_reps = sequence_output[:, text_start:text_end, :] # (1, TextLen, D)
        
        # Extract Prompt Reps
        prompt_reps_seq = sequence_output[:, 1:1+prompts_len_total, :]
        prompt_reps_reshaped = prompt_reps_seq.view(1, num_labels, PROMPT_LEN, embed_dim)
        prompt_vectors = prompt_reps_reshaped.mean(dim=2) # (1, NumLabels, D)
        
        # Similarity
        H_text = F.normalize(text_reps, dim=-1)
        H_prompts = F.normalize(prompt_vectors, dim=-1)
        
        # Logits
        logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) # (1, TextLen, NumLabels)
        
        preds = logits.argmax(-1).squeeze(0).cpu().tolist() # (TextLen)
        
        # Store
        for p, t in zip(preds, labels):
            if t != -100:
                y_true.append(t)
                y_pred.append(p)
        
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

# ==========================================================
# 6️⃣ METRICS & SAVING
# ==========================================================
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="macro", zero_division=0
)
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=relevant_label_ids, average="micro", zero_division=0
)

print(f"\n🏆 GLOBAL METRICS (No O Class):")
print(f"   • MACRO: F1={macro_f1:.4f}")
print(f"   • MICRO: F1={micro_f1:.4f}")

class_report = classification_report(
    y_true, y_pred, target_names=relevant_label_names, labels=relevant_label_ids, zero_division=0
)
print(class_report)

# Export
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{TEST_RESULTS_DIR}/eval_mono_{timestamp}.md"

# Format Output for Shell Script Parser (**Macro F1** etc)
with open(filename, "w", encoding="utf-8") as f:
    
    # Extract config
    config = checkpoint.get('config', {})
    
    # Format Config Table
    config_table = "| Parameter | Value |\n|---|---|\n"
    sorted_keys = sorted(config.keys())
    for k in sorted_keys:
        val = config[k]
        config_table += f"| **{k}** | `{val}` |\n"

    f.write(f"## 🔧 Training Configuration\n")
    f.write(f"{config_table}\n\n")

    f.write(f" ## Metriche Chiave\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| **Macro F1** | {macro_f1:.4f} |\n")
    f.write(f"| **Micro F1** | {micro_f1:.4f} |\n\n")
    
    f.write(f"## Report\n```\n{class_report}\n```\n")

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
    # Try finding it in current dir or parent
    if os.path.exists("orderTests.sh"):
         subprocess.run(["bash", "orderTests.sh"], check=True)
    elif os.path.exists("../embedding_injection/orderTests.sh"):
         subprocess.run(["bash", "../embedding_injection/orderTests.sh"], check=True)
    else:
        print("⚠️ Order script not found.")
