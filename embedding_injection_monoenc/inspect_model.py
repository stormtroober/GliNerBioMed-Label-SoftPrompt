from gliner import GLiNER
import torch

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
print(model)

# Let's inspect the underlying model
print("\n--- Model attributes ---")
print(dir(model))

if hasattr(model, 'model'):
    print("\n--- Inner Model ---")
    print(model.model)
    print(type(model.model))

# Check for encoder
if hasattr(model, 'encoder'):
    print("\n--- Encoder ---")
    print(model.encoder)
