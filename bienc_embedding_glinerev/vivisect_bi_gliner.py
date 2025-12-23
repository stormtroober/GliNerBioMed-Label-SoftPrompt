
import torch
from gliner import GLiNER
from transformers import AutoTokenizer

# Load the model
model_name = "Ihor/gliner-biomed-bi-small-v1.0"
print(f"Loading model: {model_name}")
model = GLiNER.from_pretrained(model_name)

# Get the label encoder and its tokenizer
core = model.model
lbl_enc = core.token_rep_layer.labels_encoder.model
tokenizer = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

print(f"\nLabel Encoder Model: {lbl_enc.config._name_or_path}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Check for special tokens
special_tokens_map = tokenizer.special_tokens_map
print(f"\nSpecial Tokens Map: {special_tokens_map}")

ent_token = "[ENT]"
sep_token = "[SEP]" # Standard SEP but user says there is an extra one or custom usage?

# Check if [ENT] is in vocab
if ent_token in tokenizer.vocab:
    ent_id = tokenizer.vocab[ent_token]
    print(f"'{ent_token}' ID: {ent_id}")
else:
    # It might be an added token not in standard vocab but added to tokenizer
    # Let's check all special tokens
    print(f"'{ent_token}' not in standard vocab dict. Checking added tokens...")
    # Sometimes it's added as a special token
    print(f"Additional special tokens: {tokenizer.additional_special_tokens}")
    if ent_token in tokenizer.get_vocab():
         print(f"Found in get_vocab(): {tokenizer.get_vocab()[ent_token]}")

# Function to hook and capture input_ids to label encoder
captured_input_ids = []

def hook_fn(module, args, kwargs):
    if 'input_ids' in kwargs:
        captured_input_ids.append(kwargs['input_ids'])
    elif args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.long:
        captured_input_ids.append(args[0])

handle = lbl_enc.register_forward_pre_hook(hook_fn, with_kwargs=True)

# Run a simple prediction to trigger the label encoder
text = "Alice works at Apple."
labels = ["person", "organization"]
print(f"\nRunning prediction with text: '{text}' and labels: {labels}")
model.predict_entities(text, labels)

# Analyze captured inputs
if captured_input_ids:
    print("\n--- Captured Inputs to Label Encoder ---")
    inp = captured_input_ids[0]
    print(f"Input shape: {inp.shape}")
    print(f"Input IDs: {inp.tolist()}")
    
    # Decode
    print("\nDecoded sequences:")
    for i, seq in enumerate(inp):
        decoded = tokenizer.decode(seq)
        print(f"Sequence {i}: {decoded}")
        print(f"IDs {i}: {seq.tolist()}")
        
        # Analyze breakdown
        tokens = tokenizer.convert_ids_to_tokens(seq)
        print(f"Tokens {i}: {tokens}")
else:
    print("No input_ids captured on Label Encoder.")

# Hook for Text Encoder
print("\n--- Investigating Text Encoder ---")
# Usually logic is in model.model.token_rep_layer.text_encoder? Or just model.model?
# Let's inspect model structure slightly more
print(model)

# Assuming bi-encoder has a separate text encoder
# In GLiNER class, looking at existing code might help, but let's try to hook model.model if possible
# or model.model.token_rep_layer (if it separates there)

# Based on previous explore output:
# GLiNER model has .model which is likely the BiEncoder or GlinerModel
# Let's try to hook the core model's forward
captured_text_inputs = []
def text_hook_fn(module, args, kwargs):
    print("Text Encoder Hook Triggered")
    if 'input_ids' in kwargs:
        captured_text_inputs.append(kwargs['input_ids'])
    elif args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.long:
        captured_text_inputs.append(args[0])

# Try to find text encoder
# Usually core.token_rep_layer has text_encoder if it has labels_encoder
if hasattr(core, 'token_rep_layer') and hasattr(core.token_rep_layer, 'bert_layer'):
    # In some implementations text encoder is just the transformer
    text_enc = core.token_rep_layer.bert_layer
    print("Found 'bert_layer' (likely Text Encoder)")
    text_enc.register_forward_pre_hook(text_hook_fn, with_kwargs=True)
elif hasattr(core, 'token_rep_layer') and hasattr(core.token_rep_layer, 'text_encoder'):
     text_enc = core.token_rep_layer.text_encoder
     print("Found 'text_encoder'")
     text_enc.register_forward_pre_hook(text_hook_fn, with_kwargs=True)
else:
    print("Could not locate designated text encoder sub-module directly. Hooking token_rep_layer.")
    core.token_rep_layer.register_forward_pre_hook(text_hook_fn, with_kwargs=True)


# Run prediction again
print(f"\nRunning prediction 2 with text: '{text}'")
model.predict_entities(text, labels)

if captured_text_inputs:
    print("\n--- Captured Inputs to Text Encoder ---")
    inp = captured_text_inputs[0] # The first capture might be text if labels are separate
    # Note: since we run predict twice, and we hooked label encoder previously, we might see label encoder calls again.
    # But this hook is for text encoder.
    
    print(f"Input shape: {inp.shape}")
    print(f"Input IDs: {inp.tolist()}")
    
    # Decode
    print("\nDecoded Text sequence:")
    for i, seq in enumerate(inp):
        decoded = tokenizer.decode(seq)
        print(f"Sequence {i}: {decoded}")
else:
    print("No input_ids captured on Text Encoder path.")


