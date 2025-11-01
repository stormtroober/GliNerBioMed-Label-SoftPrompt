# -*- coding: utf-8 -*-


import torch

from gliner import GLiNER
model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")

#Debug per vedere le caratteristiche del modello
# print(type(model))
# print(getattr(model, "model", None))  # molti checkpoint incapsulano sotto .model

# # ispeziona gli attributi per trovare gli encoder
# core = getattr(model, "model", model)  # fallback
# for name in dir(core):
#     if "encoder" in name.lower() or "label" in name.lower() or "proj" in name.lower():
#         try:
#             print(name, getattr(core, name).__class__)
#         except:
#             print(name, "<unavailable>")

"""Accesso ai due encoder"""

core = model.model  # SpanModel

# text encoder (DeBERTaV2, 768d)
txt_enc = core.token_rep_layer.bert_layer.model
print("Text enc hidden:", txt_enc.config.hidden_size)  # atteso 768

# label encoder (BERT 6-layer, 384d) + proiezione 384→768
lbl_enc = core.token_rep_layer.labels_encoder.model
print("Label enc hidden:", lbl_enc.config.hidden_size)  # atteso 384
print("labels_projection:", core.token_rep_layer.labels_projection)

"""Congela il Text-Encoder: non verrà aggiornato in training"""

for p in txt_enc.parameters():
    p.requires_grad = False

sum(p.requires_grad for p in txt_enc.parameters())  # deve stampare 0

"""Definisco un set di descrizioni giocattolo e il loro id"""

labels = [
    "a drug used as medication",
    "a disease or medical condition",
    "an adverse drug event or side effect"
]
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(labels)

"""Tokenizzazione delle descrizioni"""

from transformers import AutoTokenizer

lbl_name = lbl_enc.config._name_or_path
lbl_tok = AutoTokenizer.from_pretrained(lbl_name)
batch = lbl_tok(labels, return_tensors="pt", padding=True, truncation=True)

"""Ottengo embedding iniziali (“hard”)"""

from transformers import AutoTokenizer
import torch.nn.functional as F

with torch.no_grad():
    out = lbl_enc(**batch)   # out.last_hidden_state: [B,L,384]

attn = batch["attention_mask"].float()                  # [B,L]
mask = attn.unsqueeze(-1)                               # [B,L,1]
sum_emb = (out.last_hidden_state * mask).sum(dim=1)     # [B,384]
len_emb = mask.sum(dim=1).clamp(min=1e-9)               # [B,1]
HE_384 = sum_emb / len_emb                              # [B,384]  <-- come GLiNER

"""Creo la tabella "Soft" e la inizializzo"""

import torch.nn as nn

hidden_label = lbl_enc.config.hidden_size  # 384
soft_table = nn.Embedding(num_labels, hidden_label)

with torch.no_grad():
    soft_table.weight.copy_(HE_384)

"""Registrazione tabella nel modello

"""

core.token_rep_layer.soft_label_embeddings = soft_table
for p in core.token_rep_layer.labels_encoder.parameters():
    p.requires_grad = False

"""Verifica che gli embedding soft abbiano forma corretta (384) e che la proiezione li porti a 768, come richiesto dal modello."""

ids = torch.arange(num_labels)
soft_HE_384 = core.token_rep_layer.soft_label_embeddings(ids)
print(soft_HE_384.shape)  # [num_labels, 384]

# opzionale: proietta a 768 con la loro labels_projection (come fa il modello)
proj = core.token_rep_layer.labels_projection
soft_HE_768 = proj(soft_HE_384)           # [num_labels, 768]
print(soft_HE_768.shape)

"""Crea un adapter che sostituisce il label encoder con la tabella soft, aggiungendo anche parametri scalari (logit_scale, bias)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

VANILLA_LBL = lbl_enc

class _SoftLabelHF(nn.Module):
    def __init__(self, soft_table: nn.Embedding, init_logit_scale=1.0):
        super().__init__()
        self.soft = soft_table
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.bias_384 = nn.Parameter(torch.zeros(soft_table.embedding_dim))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is not None:
            B, seq_len = attention_mask.shape
            device = attention_mask.device
        else:
            B = self.soft.num_embeddings; seq_len = 1
            device = self.soft.weight.device

        if B != self.soft.num_embeddings:
            raise ValueError("Label batch size/order must match soft_table.")

        base = self.soft.weight.to(device)                         # [B,384]
        base = base * self.logit_scale + self.bias_384            # scale + bias
        last = base.unsqueeze(1).expand(B, seq_len, base.size(-1))# [B,seq,384]
        return BaseModelOutputWithPooling(last_hidden_state=last, pooler_output=base)


soft_adapter = _SoftLabelHF(soft_table)

def use_vanilla():
    core.token_rep_layer.labels_encoder.model = VANILLA_LBL
    for p in core.token_rep_layer.labels_projection.parameters():
        p.requires_grad = False  # opzionale: coerenza 100% vanilla

def use_soft(adapter):
    core.token_rep_layer.labels_encoder.model = adapter
    for p in core.token_rep_layer.labels_projection.parameters():
        p.requires_grad = True   # alleniamo la proiezione con la tabella

def compare_soft_vs_vanilla(model, texts, labels, threshold=0.2):
    # VANILLA
    use_vanilla()
    vanilla = [model.predict_entities(t, labels=labels, threshold=threshold) for t in texts]
    # SOFT
    use_soft(soft_adapter)
    soft = [model.predict_entities(t, labels=labels, threshold=threshold) for t in texts]
    for t, v, s in zip(texts, vanilla, soft):
        print("\nTEXT:", t)
        print("VANILLA:", v)
        print("SOFT   :", s)

texts = [
    "Aspirin is used to treat headaches.",
    "Ibuprofen may cause nausea."
]
gliner_labels = labels  # IMPORTANTISSIMO: stesso ordine della soft_table
compare_soft_vs_vanilla(model, texts, gliner_labels, threshold=0.2)

"""Gli score sono i medesimi perchè sto passando degli embedding di descrizioni praticamente uguali a quelli che usa il modello normalmente, dato che non ho fatto nessun training sulla Loss.

In questo blocco alleniamo la tabella di embedding "soft" delle descrizioni.

Procedimento:
1. Dataset di esempio:
   Ogni voce è (testo, (start,end), gold_label_id), dove (start,end) sono gli indici
   token dello span annotato come entità (gold) e gold_label_id è l'etichetta corretta.

2. Estrazione dello span embedding:
   - Il text encoder (congelato) produce embedding per tutti i token.
   - Facciamo la media dei token compresi tra start e end per ottenere
     lo span embedding (768d).
   - Questo rappresenta "come il modello vede" l'entità nel testo.

3. Embedding delle label:
   - Ogni label ha un embedding nella soft_table (384d).
   - Gli embedding vengono proiettati a 768d con labels_projection,
     per poter essere confrontati con lo span.

4. Calcolo dei logit:
   - Facciamo il prodotto scalare span_vec @ label_vecs.T → vettore di logit
     (uno score per ciascuna label).

5. Loss:
   - Applichiamo CrossEntropyLoss confrontando i logit con la label gold.
   - In questo modo il modello impara a far crescere lo score della label corretta
     e a ridurre quelli delle altre.

6. Ottimizzazione:
   - Aggiorniamo solo i parametri della soft_table e di labels_projection.
   - Il text encoder rimane congelato.

7. Predizione:
   - Attiviamo la modalità "soft" e chiamiamo model.predict_entities sui testi.
   - Le predizioni ora usano gli embedding soft aggiornati.
"""

train_data = [
    ("Aspirin is used to treat headaches.", (0, 0), label2id["a drug used as medication"]),
    ("Aspirin is used to treat headaches.", (5, 5), label2id["a disease or medical condition"]),
    ("Ibuprofen may cause nausea.", (0, 0), label2id["a drug used as medication"]),
    ("Ibuprofen may cause nausea.", (3, 3), label2id["an adverse drug event or side effect"]),
]


print(train_data)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(
    core.token_rep_layer.soft_label_embeddings.parameters(),
    lr=1e-3
)

txt_name = txt_enc.config._name_or_path   # txt_enc è il tuo text encoder già estratto
txt_tok  = AutoTokenizer.from_pretrained(txt_name)

def get_span_vec(text, span, tokenizer, encoder):
    """Estrae l'embedding di uno span gold dal text encoder congelato"""
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = encoder(**{k: enc[k] for k in ["input_ids","attention_mask"]})
    hidden = out.last_hidden_state.squeeze(0)  # [L,768]
    s, e = span
    return hidden[s:e+1].mean(dim=0)  # media token nello span

for epoch in range(5):
    total_loss = 0
    for text, span, gold_id in train_data:
        # 1. Span embedding dal testo
        span_vec = get_span_vec(text, span, txt_tok, txt_enc)  # [768]

        # 2. Label embeddings soft + proiezione a 768
        label_vecs = core.token_rep_layer.soft_label_embeddings.weight  # [num_labels,384]
        label_vecs = core.token_rep_layer.labels_projection(label_vecs) # [num_labels,768]

        # 3. Similarità span–label = dot product
        logits = span_vec @ label_vecs.T  # [num_labels]

        # 4. Loss cross-entropy
        target = torch.tensor([gold_id])
        loss = criterion(logits.unsqueeze(0), target)

        # 5. Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss={total_loss:.4f}")

use_soft(soft_adapter)  # attiva la tabella soft
texts = [
    "Aspirin is used to treat headaches.",
    "Ibuprofen may cause nausea."
]
preds = [model.predict_entities(t, labels=labels, threshold=0.2) for t in texts]
for t, p in zip(texts, preds):
    print("\nTEXT:", t)
    print("PRED :", p)