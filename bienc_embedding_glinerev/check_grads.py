
import torch
import torch.nn as nn
from gliner import GLiNER
from train_bi_softprompt import MLPPromptEncoder, SoftPromptLabelEncoderWrapper

MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
PROMPT_LEN = 16

print("📦 Loading model...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model
lbl_enc_model = core.token_rep_layer.labels_encoder.model

print("🔍 Checking initial parameters in labels_encoder...")
# Check if 'model' is in _modules
if 'model' in core.token_rep_layer.labels_encoder._modules:
    print("✅ 'model' is present in labels_encoder._modules")
else:
    print("❌ 'model' is NOT in labels_encoder._modules (Monkey-patching might fail registration)")

# Init Encoder
original_embeddings = lbl_enc_model.embeddings.word_embeddings
mlp_prompt_encoder = MLPPromptEncoder(
    original_embeddings=original_embeddings,
    vocab_size=original_embeddings.num_embeddings,
    embed_dim=original_embeddings.embedding_dim,
    prompt_len=PROMPT_LEN,
    pooling_mode="conv1d"
)

# Wrap
print("🛠️  Wrapping...")
wrapped_encoder = SoftPromptLabelEncoderWrapper(lbl_enc_model, mlp_prompt_encoder)
core.token_rep_layer.labels_encoder.model = wrapped_encoder

# Verify registration
print("🔍 Verifying registration after patching...")
if core.token_rep_layer.labels_encoder.model is wrapped_encoder:
    print("✅ Attribute .model updated correctly")
else:
    print("❌ Attribute .model detection FAILED")

# Check if parameters are in model.named_parameters()
found_prompt_params = False
for name, param in model.named_parameters():
    if "prompt_encoder" in name:
        found_prompt_params = True
        break

if found_prompt_params:
    print("✅ Prompt parameters found in model.named_parameters()")
else:
    print("❌ Prompt parameters NOT found in model.named_parameters()")

# Freeze check
print("❄️  Applying freezing logic...")
for param in model.parameters():
    param.requires_grad = False
for param in core.token_rep_layer.labels_encoder.model.prompt_encoder.parameters():
    param.requires_grad = True

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"🔥 Trainable params count: {len(trainable)}")
if len(trainable) > 0:
    print(f"Example trainable: {trainable[0]}")
else:
    print("❌ No trainable parameters found!")

# Dummy Forward/Backward
print("🏃 Running Dummy Train Step...")
model.train()
# Prepare inputs
text = "Sample text"
labels = ["label1", "label2"]
# We need to construct batch manually to call forward or use model.forward directly?
# model.forward expects input_ids usually. GLiNER forward is complex.
# We can use predict_entities logic or manually construct inputs.
# Better: use model.train_model logic on a tiny batch.

# Or just verify gradient flow through the wrapped sub-module directly.
# This avoids full model complexity but verifies the wrapper.

# Inputs for wrapper
# labels are tokenized: [CLS] label [SEP]
input_ids = torch.tensor([[101, 2000, 102], [101, 2001, 102]]) # Batch=2
attn_mask = torch.ones_like(input_ids)

# Run wrapper forward
print("   Wrapper Forward...")
output = wrapped_encoder(input_ids=input_ids, attention_mask=attn_mask)
# output is likely BaseModelOutput or similar (tuple or dataclass)
if hasattr(output, 'last_hidden_state'):
    res = output.last_hidden_state
else:
    res = output[0]

print(f"   Output shape: {res.shape}")
loss = res.mean()
print("   Backward...")
loss.backward()

# Check gradients
print("🔍 Checking Gradients...")
has_grad = False
for name, param in mlp_prompt_encoder.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"   Param {name}: Grad Norm = {grad_norm}")
        if grad_norm > 0:
            has_grad = True
    else:
        print(f"   Param {name}: Grad is None!")

if has_grad:
    print("✅ Gradients are flowing to MLPPromptEncoder!")
else:
    print("❌ Gradients are NOT flowing (all zero or None).")

