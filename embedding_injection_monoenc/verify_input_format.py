from gliner import GLiNER
import torch

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
tokenizer = model.data_processor.transformer_tokenizer

print("\n--- Model Config ---")
print(model.config)

print("\n--- Inner Model Config ---")
if hasattr(model, 'model'):
    print(model.model.config)

print("\n--- Tokenizer ---")
print(tokenizer)
print(f"CLS: {tokenizer.cls_token_id}, SEP: {tokenizer.sep_token_id}, PAD: {tokenizer.pad_token_id}")

print("\n--- Special Tokens ---")
print(tokenizer.special_tokens_map)
print(f"Additional: {tokenizer.additional_special_tokens}")

if hasattr(model.config, 'class_token_index'):
    print(f"Class token index: {model.config.class_token_index}")
elif hasattr(model.model.config, 'class_token_index'):
    print(f"Class token index (inner): {model.model.config.class_token_index}")

# Try to see how inputs are tokenized
text = "Alice lives in Wonderland"
labels = ["person", "location"]

# We can mimic what likely happens using the tokenizer
# Standard GLiNER (UniEncoder) usually does: [CLS] <<person>> <<location>> [SEP] Alice lives in Wonderland [SEP]
# where <<class>> is a special token, or it just uses token ids.

# Let's inspect the vocab to see if there are special tokens like <<ENT>> or similar?
# Or maybe it uses a specific token like `<<label>>`?

# Let's print the first few special tokens from the tokenizer if possible
# or just check if `<<` tokens exist.
print("Checking for '<<' tokens:")
for t in tokenizer.additional_special_tokens:
    print(t)

