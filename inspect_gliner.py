
import torch
from gliner import GLiNER

model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
print("Labels Encoder Type:", type(model.model.token_rep_layer.labels_encoder))
print("Labels Encoder Dir:", dir(model.model.token_rep_layer.labels_encoder))
try:
    print("Labels Encoder Config:", model.model.token_rep_layer.labels_encoder.config)
except:
    pass
