import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import random

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
        if hidden_dim is None: 
            hidden_dim = embed_dim * 4
            
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        return x

class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, label2id):
        print(f"📖 Leggendo {path_json}...")
        with open(path_json, "r", encoding="utf-8") as f: 
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        
    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Tokenize text
        input_ids = self.tok.convert_tokens_to_ids(rec["tokens"])
        
        # Determine labels present in this example for "soft prompt" construction
        # Note: In mono-encoder, we usually provide ALL potential labels or a subset.
        # For simplicity and coverage, we can provide ALL labels as prompts every time,
        # or sample negatives.
        # Let's start with ALL labels to be safe and consistent.
        
        # We need to construct the input such that:
        # [CLS] [Prompt1] [Prompt2] ... [SEP] [Text Tokens] [SEP]
        
        # However, the dataset just returns raw data. The Collator will handle the complex construction.
        return {
            "token_ids": input_ids,
            "labels": rec["labels"], # List of label IDs for each token
        }

def collate_fn_mono(batch, tokenizer, label2id, device="cpu", prompt_len=1):
    """
    Custom collator for Mono-Encoder with Soft Prompt Injection.
    Constructs inputs: [CLS] [P_1] .. [P_N] [SEP] [Text] [SEP]
    """
    
    # 1. Prepare Text Batch
    # Find max text length
    max_text_len = max(len(x["token_ids"]) for x in batch)
    
    # 2. Prepare Prompts
    # We use all labels as prompts
    # label_names = list(label2id.keys())
    # label_ids_list = list(label2id.values())
    
    # In this specific task, we want to inject embeddings.
    # So we need to return the IDs of the labels so the MLPPromptEncoder can generate embeddings.
    # And we need to prepare the "text" part.
    
    # Let's simplify: All examples get ALL labels.
    all_label_ids = torch.tensor(list(label2id.values()), device=device) # (NumClasses)
    num_classes = len(all_label_ids)
    
    # Calculate Total Length: 1 (CLS) + NumClasses*1 (Prompts) + 1 (SEP) + TextLen + 1 (SEP)
    # Actually GLiNER formatting is usually: [CLS] P1 P2 ... [SEP] T1 T2 ... [SEP]
    
    B = len(batch)
    
    # Create placeholders
    # We will return:
    # - text_input_ids: (B, MaxTextLen) -> to be embedded by BERT
    # - label_input_ids: (B, NumClasses) -> to be embedded by MLP
    # - attention_mask: (B, TotalSeqLen)
    # - input_ids_dummy: (B, TotalSeqLen) -> Just for passing to functions that check shape
    
    text_input_ids = torch.full((B, max_text_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels_tensor = torch.full((B, max_text_len), -100, dtype=torch.long, device=device)
    
    # Span/Entity masking logic might be needed if using Span Loss, 
    # but user code used standard Token Classification (CrossEntropy/Focal) on "labels".
    # Wait, the previous code used "GLiNER" model but treated it as token classification?
    # Let's check `glinerbiomed_mlp_focal_cbclass_prompt_validation.py`.
    # It calculates logits: `logits = torch.matmul(H_text, label_matrix.T)`
    # This is effectively checking each token against each label embedding.
    # So it IS token classification (or rather, token-label similarity).
    
    # In Mono-Encoder, we concatenate. The output of the Transformer will be (B, SeqLen, Dim).
    # We need to extract the representations of the Text Tokens and the Prompt Tokens.
    
    prompts_start_idx = 1 # After [CLS]
    text_start_idx = 1 + num_classes + 1 # [CLS] P...P [SEP]
    
    total_len = text_start_idx + max_text_len + 1 # + [SEP]
    
    input_ids_final = torch.full((B, total_len), tokenizer.pad_token_id, device=device)
    attention_mask = torch.zeros((B, total_len), device=device)
    
    # Fill standard tokens
    input_ids_final[:, 0] = tokenizer.cls_token_id # CLS
    
    # We don't have "tokens" for prompts in input_ids_final because they are soft injected.
    # But we need to put SOMETHING there so the model doesn't crash if it checks ranges.
    # Usually we can put [MASK] or a special token, or just PAD (but mask=1).
    # Let's ensure attention_mask is 1.
    
    # Set separators
    input_ids_final[:, 1 + num_classes] = tokenizer.sep_token_id # SEP after prompts
    
    for i, ex in enumerate(batch):
        L = len(ex["token_ids"])
        
        # Text input ids (for embedding lookup)
        text_input_ids[i, :L] = torch.tensor(ex["token_ids"], device=device)
        
        # Final input IDs (Text part)
        col_start = text_start_idx
        col_end = col_start + L
        input_ids_final[i, col_start:col_end] = torch.tensor(ex["token_ids"], device=device)
        input_ids_final[i, col_end] = tokenizer.sep_token_id # SEP after text
        
        # Attention Mask
        # 1 for CLS
        # 1 for Prompts
        # 1 for SEP
        # 1 for Text
        # 1 for final SEP
        attention_mask[i, :col_end+1] = 1
        
        # Labels (aligned with text)
        labels_tensor[i, :L] = torch.tensor(ex["labels"], device=device)

    # Label IDs for MLP (same for all batch elements usually, but let's replicate)
    label_input_ids_batch = all_label_ids.unsqueeze(0).expand(B, -1)
    
    return {
        "text_input_ids": text_input_ids,       # (B, T)
        "label_input_ids": label_input_ids_batch, # (B, C)
        "input_ids_final": input_ids_final,     # (B, S_total) Reference
        "attention_mask": attention_mask,       # (B, S_total)
        "labels": labels_tensor,                # (B, T)
        "prompt_indices": (prompts_start_idx, prompts_start_idx+num_classes),
        "text_indices": (text_start_idx, text_start_idx+max_text_len) # Note: max len, not dynamic
    }
