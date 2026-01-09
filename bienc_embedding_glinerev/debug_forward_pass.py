
import torch
import json
import os
from transformers import AutoTokenizer
from gliner import GLiNER
# Import classes from training script to replicate environment
# We will copy-paste the relevant classes to ensure standalone execution
import torch.nn as nn

# --- COPIA CLASSI MODIFICATE ---
class PromptPooler(nn.Module):
    def __init__(self, embed_dim, prompt_len, mode="adaptive_avg", max_seq_len=512):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        if mode == "conv1d": # Simplified for brevity, matching default
             self.conv_layers = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
             self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
             self.norm = nn.LayerNorm(embed_dim)
        else: # Fallback or other modes
             self.pooler = nn.AdaptiveAvgPool1d(prompt_len)
    
    def forward(self, x, attention_mask=None):
        if self.mode == "conv1d":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = x * mask_expanded
            x_t = x.transpose(1, 2)
            conv_out = self.conv_layers(x_t)
            pooled = self.adaptive_pool(conv_out)
            return self.norm(pooled.transpose(1, 2))
        return x # Placeholder

class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim, hidden_dim=None, dropout=0.1, prompt_len=32, pooling_mode="conv1d"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        if hidden_dim is None: hidden_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.norm(x + self.mlp(x))
        if self.pooler: x = self.pooler(x, attention_mask)
        return x

class SoftPromptLabelEncoderWrapper(nn.Module):
    def __init__(self, original_encoder, prompt_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.prompt_encoder = prompt_encoder
        self.config = original_encoder.config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        print(f"  [DEBUG WRAPPER] Input IDs shape: {input_ids.shape}")
        
        soft_prompts = self.prompt_encoder(input_ids, attention_mask)
        print(f"  [DEBUG WRAPPER] Soft Prompts shape: {soft_prompts.shape}")
        
        std_embeddings_layer = self.original_encoder.embeddings
        
        # Estrai CLS e SEP
        # Nota: Qui assumiamo che input_ids contenga i token delle DESCRIZIONI
        token_embeddings = std_embeddings_layer(input_ids=input_ids)
        cls_embeds = token_embeddings[:, 0:1, :]
        
        sep_token_id = getattr(self.config, 'sep_token_id', 2)
        sep_ids = torch.full((input_ids.shape[0], 1), sep_token_id, device=input_ids.device, dtype=torch.long)
        sep_embeds = std_embeddings_layer(input_ids=sep_ids)
        
        final_embeds = torch.cat([cls_embeds, soft_prompts, sep_embeds], dim=1)
        print(f"  [DEBUG WRAPPER] Final Embeds (Sandwich) shape: {final_embeds.shape}")

        B = input_ids.shape[0]
        P = soft_prompts.shape[1]
        final_mask = torch.ones((B, 1 + P + 1), device=input_ids.device, dtype=attention_mask.dtype)
        
        outputs = self.original_encoder(inputs_embeds=final_embeds, attention_mask=final_mask, **kwargs)
        return outputs
    
    def __getattr__(self, name):
         if name in ["original_encoder", "prompt_encoder", "config", "forward"]: return super().__getattr__(name)
         return getattr(self.original_encoder, name)

# --- DEBUG SCRIPT ---
def debug_main():
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
    print(f"Loading {MODEL_NAME}...")
    model = GLiNER.from_pretrained(MODEL_NAME)
    
    # Setup Components
    core = model.model
    lbl_enc_model = core.token_rep_layer.labels_encoder.model
    original_embeddings = lbl_enc_model.embeddings.word_embeddings
    
    prompt_enc = MLPPromptEncoder(original_embeddings, original_embeddings.num_embeddings, original_embeddings.embedding_dim)
    wrapped = SoftPromptLabelEncoderWrapper(lbl_enc_model, prompt_enc)
    core.token_rep_layer.labels_encoder.model = wrapped
    
    # Patch Encode Labels (Manual logic replay)
    import types
    def patched_encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        print(f"  [DEBUG PATCH] Called encode_labels. Input shape: {input_ids.shape}")
        
        prompt_len = 32 # Hardcoded for debug
        batch_size = input_ids.shape[0]
        total_len = prompt_len + 2
        pooling_mask = torch.ones((batch_size, total_len), device=input_ids.device, dtype=attention_mask.dtype)
        
        label_kwargs = dict(kwargs)
        # Clean kwargs
        label_kwargs.pop("packing_config", None); label_kwargs.pop("pair_attention_mask", None)
        label_kwargs["attention_mask"] = attention_mask 

        # Forward
        out = self.labels_encoder(input_ids, *args, **label_kwargs)
        if hasattr(out, "last_hidden_state"): out = out.last_hidden_state
        if hasattr(self, "labels_projection"): out = self.labels_projection(out)
        
        print(f"  [DEBUG PATCH] Output BEFORE pooling: {out.shape} (Expect B, {total_len}, D)")
        
        pooled = self.mean_pooling(out, pooling_mask)
        print(f"  [DEBUG PATCH] Pooled output: {pooled.shape}")
        return pooled

    core.token_rep_layer.encode_labels = types.MethodType(patched_encode_labels, core.token_rep_layer)
    
    # Create Dummy Batch
    # GLiNER train_model expects a dataset list.
    # We simulate what happens inside the collator.
    
    # 1. Labels: List of descriptions
    labels = ["A protein involved in immune response.", "Small molecule drug."]
    print(f"\nSimulating Labels: {labels}")
    
    # We need to Tokenize these labels using the LABEL tokenizer
    # Because GLiNER does this internally? Or does it expect raw strings?
    # GLiNER's data collator usually receives strings and tokenizes them on the fly if needed
    # OR it pre-tokenizes?
    
    # Let's inspect how we are passing data in train script.
    # We pass 'data' list where 'ner' has description strings.
    # [start, end, "DescriptionString"]
    
    # GLiNER model.train_model(...) -> internally creates a data collator.
    # The collator takes ALL unique labels in the batch.
    
    # Let's try to run a manual forward pass on the Model Logic directly.
    # The BiEncoder logic:
    # 1. Text Encoding
    # 2. Label Encoding (of ALL unique labels)
    
    # Simulate Label Encoding Step
    # We need the tokenizer for labels
    # Use the one from the model
    # Wait, GLiNER class encapsulates tokenizers.
    
    # NOTE: GLiNER.train_model is a high level wrapper.
    # We want to check if the INPUT to wrapped_encoder is correct.
    
    # Let's fake 'input_ids' for the wrapper.
    # The label tokenizer is likely 'bert-base-cased' or similar (biomed specific)
    # Access via private attribute if possible or reload.
    # Actually, let's just use the model's loaded config to guess or reuse 'lbl_enc_model' tokenizer?
    # It's hard to extract the tokenizer instance from the GLiNER object easily without private access.
    # Let's assume standard tokenizer for now just to create dummy IDs.
    
    vocab_size = original_embeddings.num_embeddings
    dummy_input_ids = torch.randint(0, vocab_size, (2, 15)) # 2 labels, length 15
    dummy_mask = torch.ones((2, 15))
    
    print("\n--- RUNNING FORWARD PASS ---")
    # Call encode_labels (which hits our Patch -> Wrapper)
    encoded_labels = core.token_rep_layer.encode_labels(dummy_input_ids, dummy_mask)
    
    print(f"\n✅ Result Shape: {encoded_labels.shape}")
    print("If this shape is (2, hidden_dim), then the forward pass structure is correct.")

if __name__ == "__main__":
    debug_main()
