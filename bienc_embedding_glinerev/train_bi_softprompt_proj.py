import os
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from gliner import GLiNER
from transformers import AutoTokenizer, TrainingArguments

# ==========================================================
# 🔧 PARAMETERS
# ==========================================================
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
DATASET_PATH = "../dataset/dataset_span_bi.json" # Adjust path if on Kaggle
LABEL2ID_PATH = "../dataset/label2id.json"
LABEL2DESC_PATH = "../dataset/label2desc.json"
PROMPT_LEN = 32 #8, 16, 32
POOLING_MODE = "conv1d" # "adaptive_avg", "adaptive_max", "attention", "conv1d", "conv1d_strided"
MLP_LR = 1e-4 # Learning rate for MLP
WEIGHT_DECAY = 0.01
EPOCHS = 5
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
DROPOUT_PROMPT = 0.1
FOCAL_LOSS_ALPHA = 0.4055
FOCAL_LOSS_GAMMA = 2.83

# ==========================================================
# 1️⃣ PROMPT POOLER & ENCODER
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
            # Learnable query tokens che estraggono PROMPT_LEN rappresentazioni
            self.queries = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d":
            # Conv1D downsampling: apprende come comprimere la sequenza
            self.conv_layers = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
            self.norm = nn.LayerNorm(embed_dim)
        elif mode == "conv1d_strided":
            # Versione più aggressiva con gating mechanism
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
        """
        Args:
            x: (B, seq_len, dim)
            attention_mask: (B, seq_len) - 1 per token validi, 0 per padding
        Returns:
            (B, prompt_len, dim)
        """
        B, seq_len, dim = x.shape
        
        if self.mode in ["adaptive_avg", "adaptive_max"]:
            # AdaptivePool lavora su ultima dim, quindi permute
            # (B, seq_len, dim) -> (B, dim, seq_len)
            x_t = x.transpose(1, 2)
            
            # Applica mask se presente (sostituisci padding con 0 per avg, -inf per max)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).float()  # (B, 1, seq_len)
                if self.mode == "adaptive_avg":
                    x_t = x_t * mask_expanded
                else:  # adaptive_max
                    x_t = x_t.masked_fill(mask_expanded == 0, float('-inf'))
            
            # (B, dim, seq_len) -> (B, dim, prompt_len)
            pooled = self.pooler(x_t)
            # (B, dim, prompt_len) -> (B, prompt_len, dim)
            return pooled.transpose(1, 2)
        
        elif self.mode == "attention":
            # Queries apprese: (1, prompt_len, dim) -> (B, prompt_len, dim)
            queries = self.queries.expand(B, -1, -1)
            
            # Key padding mask per attention (True = ignore)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
            
            # Cross-attention: queries attendono a x
            attn_out, _ = self.attn(queries, x, x, key_padding_mask=key_padding_mask)
            return self.norm(attn_out + queries)  # Residual
        
        elif self.mode == "conv1d":
            # Applica mask prima della convoluzione
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            
            x_t = x.transpose(1, 2)  # (B, dim, seq_len)
            conv_out = self.conv_layers(x_t)  # (B, dim, seq_len)
            pooled = self.adaptive_pool(conv_out)  # (B, dim, prompt_len)
            pooled = pooled.transpose(1, 2)  # (B, prompt_len, dim)
            return self.norm(pooled)
        
        elif self.mode == "conv1d_strided":
            # Versione con gating mechanism
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            
            x_t = x.transpose(1, 2)  # (B, dim, seq_len)
            conv_out = self.conv(x_t)  # (B, dim, seq_len)
            pooled = self.adaptive_pool(conv_out)  # (B, dim, prompt_len)
            pooled = pooled.transpose(1, 2)  # (B, prompt_len, dim)
            
            # Gating: permette al modello di decidere quanto "fidarsi" della convoluzione
            gate = self.gate(pooled)
            pooled = pooled * gate
            
            return self.norm(pooled)

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, 
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize with original word embeddings
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

# ==========================================================
# 2️⃣ WRAPPER FOR LABEL ENCODER
# ==========================================================
class SoftPromptLabelEncoderWrapper(nn.Module):
    def __init__(self, original_encoder, prompt_encoder):
        super().__init__()
        self.original_encoder = original_encoder # The BERT/Transformer model
        self.prompt_encoder = prompt_encoder     # The MLP Prompt Encoder
        self.config = original_encoder.config    # Expose config for compatibility

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 1. Generate Soft Prompts (The REPLACEMENT content)
        soft_prompts = self.prompt_encoder(input_ids, attention_mask) # (B, P, D)
        
        # 2. Get the original embeddings to extract [CLS] and [SEP]
        std_embeddings_layer = self.original_encoder.embeddings
        # minimal call to get static embeddings
        token_embeddings = std_embeddings_layer(input_ids=input_ids) # (B, L, D)
        
        # Extract [CLS] (always at index 0)
        cls_embeds = token_embeddings[:, 0:1, :] # (B, 1, D)
        
        # Create a [SEP] token embedding.
        sep_token_id = getattr(self.config, 'sep_token_id', None)
        if sep_token_id is None:
            sep_token_id = 2 # Default to 2 (DeBERTa-v3) if not found. BERT uses 102.
        sep_ids = torch.full((input_ids.shape[0], 1), sep_token_id, device=input_ids.device, dtype=torch.long)
        sep_embeds = std_embeddings_layer(input_ids=sep_ids) # (B, 1, D)
        
        # 3. Sandwich: [CLS] + [Soft Prompts] + [SEP]
        final_embeds = torch.cat([cls_embeds, soft_prompts, sep_embeds], dim=1) # (B, 1 + P + 1, D)
        
        # 4. Create Mask
        # Structure: 1 (CLS) + P (Prompts) + 1 (SEP)
        B = input_ids.shape[0]
        P = soft_prompts.shape[1]
        
        # Mask is all 1s because we constructed a valid dense sequence
        final_mask = torch.ones((B, 1 + P + 1), device=input_ids.device, dtype=attention_mask.dtype)
        
        # 5. Forward
        outputs = self.original_encoder(inputs_embeds=final_embeds, attention_mask=final_mask, **kwargs)
        
        return outputs

    def __getattr__(self, name):
         if name in ["original_encoder", "prompt_encoder", "config", "forward"]:
             return super().__getattr__(name)
         return getattr(self.original_encoder, name)

def patch_bi_encoder_for_soft_prompts(bi_encoder_module):
    """
    Patches the BiEncoder module's encode_labels method.
    Structure is now: [CLS] [Prompt_Vectors] [SEP]
    Pooling should consider this whole sequence.
    """
    import types

    def patched_encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        # 1. Determine Prompt Length
        try:
            prompt_len = self.labels_encoder.model.prompt_encoder.pooler.prompt_len
        except AttributeError:
            prompt_len = 0
            
        batch_size = input_ids.shape[0]
        
        # 2. Construct the mask expected by Mean Pooling
        # Structure: [CLS] (1) + [Prompts] (P) + [SEP] (1)
        # Total Length: P + 2
        if prompt_len > 0:
            total_len = prompt_len + 2
            pooling_mask = torch.ones((batch_size, total_len), device=input_ids.device, dtype=attention_mask.dtype)
        else:
            # Fallback if no prompt len (shouldn't happen in this setup)
            pooling_mask = attention_mask

        # 3. Call the encoder
        label_kwargs = dict(kwargs)
        label_kwargs.pop("packing_config", None)
        label_kwargs.pop("pair_attention_mask", None)
        # Pass original mask for the MLP internal processing
        label_kwargs["attention_mask"] = attention_mask 
        
        labels_embeddings = self.labels_encoder(input_ids, *args, **label_kwargs)
        
        if hasattr(labels_embeddings, "last_hidden_state"):
            labels_embeddings = labels_embeddings.last_hidden_state
            
        if hasattr(self, "labels_projection"):
            labels_embeddings = self.labels_projection(labels_embeddings)
            
        # 4. Pool
        labels_embeddings = self.mean_pooling(labels_embeddings, pooling_mask)
        return labels_embeddings

    # Apply patch
    bi_encoder_module.encode_labels = types.MethodType(patched_encode_labels, bi_encoder_module)
    print("🧩 Monkey-patch applied")

def load_and_map_labels(dataset, label2id_path, label2desc_path):
    print(f"📖 Loading label mappings from {label2id_path} and {label2desc_path}")
    with open(label2id_path) as f: label2id = json.load(f)
    with open(label2desc_path) as f: label2desc = json.load(f)
    
    # Create ID -> Description mapping
    # Note: Dataset labels are strings "0", "1", etc.
    id2desc = {}
    for label_name, idx in label2id.items():
        id2desc[str(idx)] = label2desc[label_name]
        
    print("🔄 Replacing numeric labels with descriptions...")
    count = 0
    for item in dataset:
        for ner in item['ner']:
            # ner is [start, end, label]
            original_label = ner[2]
            if original_label in id2desc:
                ner[2] = id2desc[original_label]
                count += 1
                
    print(f"✅ Replaced {count} labels with descriptions.")
    return dataset

# ==========================================================
# 3️⃣ MAIN SCRIPT
# ==========================================================
if __name__ == "__main__":
    
    print(f"📦 Loading model: {MODEL_NAME}")
    model = GLiNER.from_pretrained(MODEL_NAME)
    
    # Identify Components
    core = model.model
    # BiEncoder structure: model.token_rep_layer.labels_encoder.model is the Transformer
    # Verify we are on a BiEncoder
    if not hasattr(core, 'token_rep_layer') or not hasattr(core.token_rep_layer, 'labels_encoder'):
        raise ValueError("Model does not appear to be a standard GLiNER Bi-Encoder (missing token_rep_layer.labels_encoder)")
    
    lbl_enc_model = core.token_rep_layer.labels_encoder.model 
    
    # Identify Embeddings for MLP init
    # Verified: BERT-like structure used in Ihor/gliner-biomed-bi-small-v1.0
    # We perform a sanity check to ensure the model structure is as expected.
    if not hasattr(lbl_enc_model, 'embeddings') or not hasattr(lbl_enc_model.embeddings, 'word_embeddings'):
        raise ValueError(f"Unexpected Label Encoder structure: {type(lbl_enc_model)}. Expected BERT-like with .embeddings.word_embeddings")
    
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    vocab_size = original_embeddings.num_embeddings
    embed_dim = original_embeddings.embedding_dim
    
    print(f"✨ Initializing MLP Prompt Encoder with Prompt Length: {PROMPT_LEN}")
    mlp_prompt_encoder = MLPPromptEncoder(
        original_embeddings=original_embeddings,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        prompt_len=PROMPT_LEN,
        pooling_mode=POOLING_MODE,
        dropout=DROPOUT_PROMPT
    )
    
    # WRAP the label encoder
    print("🛠️  Wrapping Label Encoder...")
    wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, mlp_prompt_encoder)
    
    # REPLACE in the model
    # We replace the inner transformer model
    core.token_rep_layer.labels_encoder.model = wrapped_encoder
    
    # Patch the BiEncoder to handle Soft Prompt logic
    patch_bi_encoder_for_soft_prompts(core.token_rep_layer)
    
    # FREEZING STRATEGY
    print("❄️  Freezing parameters...")
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze ONLY the MLP Prompt Encoder
    print("🔓 Unfreezing MLP Prompt Encoder...")
    for param in core.token_rep_layer.labels_encoder.model.prompt_encoder.parameters():
        param.requires_grad = True

    # Unfreeze Labels Projection if available
    if hasattr(core.token_rep_layer, 'labels_projection'):
        print("🔓 Unfreezing Labels Projection...")
        for param in core.token_rep_layer.labels_projection.parameters():
            param.requires_grad = True
    else:
        print("⚠️ Warning: labels_projection not found, cannot unfreeze it.")
        
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Trainable Parameters: {trainable_params}")
    
    # Load Dataset
    print(f"📖 Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
        
    # Apply Description Mapping
    data = load_and_map_labels(data, LABEL2ID_PATH, LABEL2DESC_PATH)
    
    # Split (Simple manual split for demo)
    train_data = data[:int(len(data)*0.9)]
    eval_data = data[int(len(data)*0.9):]

    print(f'\nFinal Train dataset size: {len(train_data)}')
    print(f'Final Test dataset size: {len(eval_data)}')

    # ==========================================
    # TRAINING CONFIGURATION DISPLAY
    # ==========================================
    num_batches_per_epoch = len(train_data) // BATCH_SIZE
    total_steps = (EPOCHS * num_batches_per_epoch) // GRAD_ACCUMULATION

    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f'Mode: EPOCHS (Target: {EPOCHS})')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Gradient Accumulation: {GRAD_ACCUMULATION}')
    print(f'Steps per Epoch (Effective): {num_batches_per_epoch // GRAD_ACCUMULATION}')
    print(f'Total Steps: {total_steps}')
    print("="*50 + "\n")
    
    print("🚀 Starting Training...")
    # Using the standard train_model function
    model.train_model(
        train_dataset=train_data, 
        eval_dataset=eval_data, 
        
        # Use specific arguments for HF Trainer
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # Learning Rates
        learning_rate=MLP_LR, 
        others_lr=MLP_LR,
        
        num_train_epochs=EPOCHS,
        max_steps=total_steps,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRAD_ACCUMULATION,

        # Logging
        logging_steps=max(1, num_batches_per_epoch // 10),
        
        # Evaluation & Saving
        eval_strategy="steps", 
        save_strategy="steps",
        eval_steps=num_batches_per_epoch,
        save_steps=num_batches_per_epoch,
        output_dir="savings_softprompt_proj",
        
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Hardware
        use_cpu=not torch.cuda.is_available(),
        fp16=(torch.cuda.is_available() and not (torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False)),
        bf16=(torch.cuda.is_available() and (torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False)),
        
        # Loss
        focal_loss_alpha=FOCAL_LOSS_ALPHA,
        focal_loss_gamma=FOCAL_LOSS_GAMMA,
    )
    
    print("✅ Training Complete.")

    # ==========================================
    # SAVING CHECKPOINT (.pt)
    # ==========================================
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a LIGHTWEIGHT checkpoint dictionary (Prompts Only)
    # We navigate to the specific module we trained
    prompt_encoder_weights = model.model.token_rep_layer.labels_encoder.model.prompt_encoder.state_dict()
    
    checkpoint = {
        "prompt_encoder_state_dict": prompt_encoder_weights, # SAVE ONLY THE ADAPTER
        "training_metadata": {
            "base_model_name": MODEL_NAME,
            "train_dataset_size": len(train_data),
            "eval_dataset_size": len(eval_data),
            "num_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate (MLP)": MLP_LR, 
            "weight_decay": WEIGHT_DECAY,
            "grad_accumulation": GRAD_ACCUMULATION,
            "prompt_len": PROMPT_LEN,
            "pooling_mode": POOLING_MODE,
            "dropout_prompt": DROPOUT_PROMPT,
            "timestamp": timestamp,
            "unfrozen_components": ["prompt_encoder", "labels_projection"]
        },
        # Architecture parameters needed for reconstruction
        "architecture_params": {
            "prompt_len": PROMPT_LEN,
            "pooling_mode": POOLING_MODE,
            "dropout": DROPOUT_PROMPT,
            "hidden_dim": None, # Default used
            "embed_dim": embed_dim,
            "vocab_size": vocab_size
        }
    }
    
    # Save Label Projection if available
    if hasattr(core.token_rep_layer, 'labels_projection'):
        print("💾 Saving Labels Projection weights...")
        checkpoint["labels_projection_state_dict"] = core.token_rep_layer.labels_projection.state_dict()

    if not os.path.exists("savings"):
        os.makedirs("savings")
        
    pt_filename = f"bi_softprompt_proj_adapter_{timestamp}.pt"
    pt_path = os.path.join("savings", pt_filename)
    
    torch.save(checkpoint, pt_path)
    print(f"💾 Prompt Encoder + Projection Adapter (.pt) saved to {pt_path}")
    print("ℹ️  To use this, load the base model, and inject these weights into the MLPPromptEncoder AND the Labels Projection.")

    # Cleanup temporary checkpoints
    if os.path.exists("savings_softprompt_proj"):
        print("🧹 Cleaning up temporary checkpoints in savings_softprompt_proj...")
        shutil.rmtree("savings_softprompt_proj", ignore_errors=True)

