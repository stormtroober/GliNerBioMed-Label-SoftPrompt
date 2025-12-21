import torch
from gliner import GLiNER

MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

# Hidden size dei due encoder
txt_hidden = txt_enc.config.hidden_size
lbl_hidden = lbl_enc.config.hidden_size

print(f"text encoder hidden_size = {txt_hidden}")
print(f"label encoder hidden_size = {lbl_hidden}")
print(f"dims_match = {txt_hidden == lbl_hidden}")

# Nome/modello di origine (utile per capire se usano la stessa base)
print(f"text encoder base = {getattr(txt_enc.config, '_name_or_path', 'N/A')}")
print(f"label encoder base = {getattr(lbl_enc.config, '_name_or_path', 'N/A')}")

# Stato della projection
print("labels_projection =", proj)
if hasattr(proj, "in_features") and hasattr(proj, "out_features"):
    print(f"projection shape: {proj.in_features} -> {proj.out_features}")

# Visualizzazione ARCHITECTURE di entrambi gli encoder
print("\n" + "="*80)
print("TEXT ENCODER - ARCHITECTURE")
print("="*80)
print(txt_enc)

print("\n" + "="*80)
print("LABEL ENCODER - ARCHITECTURE")
print("="*80)
print(lbl_enc)