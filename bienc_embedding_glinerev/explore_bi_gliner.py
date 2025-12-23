
import torch
from gliner import GLiNER
from transformers import AutoTokenizer

# Load the model
model_name = "Ihor/gliner-biomed-bi-small-v1.0"
print(f"Loading model: {model_name}")
model = GLiNER.from_pretrained(model_name)

# Identify the label encoder
print("\nModel structure:")
print(model)

# Try to pinpoint the label encoder
# Based on train_mlp_focal_cb_val.py: core.token_rep_layer.labels_encoder.model
core = model.model
if hasattr(core, 'token_rep_layer'):
    print("\nFound token_rep_layer")
    if hasattr(core.token_rep_layer, 'labels_encoder'):
        print("Found labels_encoder")
        lbl_enc = core.token_rep_layer.labels_encoder.model
        print(f"Label Encoder type: {type(lbl_enc)}")
        
        # Add a hook to print input_ids when it's called
        def hook_fn(module, args, kwargs):
            print("\n--- HOOK: Label Encoder Forward ---")
            if args:
                print(f"Args[0] shape: {args[0].shape}")
                print(f"Args[0] content: {args[0]}")
            if kwargs:
                # Check for input_ids or inputs_embeds
                if 'input_ids' in kwargs:
                    print(f"Kwargs['input_ids'] shape: {kwargs['input_ids'].shape}")
                    print(f"Kwargs['input_ids']: {kwargs['input_ids']}")
                    
                    # Decode to see special tokens
                    tokenizer = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)
                    for i in range(min(3, kwargs['input_ids'].shape[0])):
                        decoded = tokenizer.decode(kwargs['input_ids'][i])
                        print(f"Decoded[{i}]: {decoded}")
                        
                if 'inputs_embeds' in kwargs:
                    print(f"Kwargs['inputs_embeds'] shape: {kwargs['inputs_embeds'].shape}")
        
        handle = lbl_enc.register_forward_pre_hook(hook_fn, with_kwargs=True)
        print("Hook registered.")

# Define inputs
text = "John works at Google"
labels = ["person", "organization"]

print(f"\nPredicting entities for text: '{text}'")
print(f"Labels: {labels}")

# Run prediction
entities = model.predict_entities(text, labels)

print("\nEntities found:")
print(entities)
