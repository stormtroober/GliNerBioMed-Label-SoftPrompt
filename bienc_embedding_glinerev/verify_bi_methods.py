
import torch
from gliner import GLiNER
import inspect

model_name = "Ihor/gliner-biomed-bi-small-v1.0"
model = GLiNER.from_pretrained(model_name)

print(f"Model class: {type(model)}")

if hasattr(model, 'get_representations'):
    print("\n✅ 'get_representations' exists.")
    print(inspect.getsource(model.get_representations))
else:
    print("\n❌ 'get_representations' DOES NOT exist on this model instance.")

# Check for other potential methods
print(f"\nModel methods: {[m for m in dir(model) if not m.startswith('_')]}")
