import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
from transformers import AutoTokenizer
from gliner import GLiNER
from sklearn.metrics import precision_recall_fscore_support

import os

def save_trained_model(model, save_dir="./trained_gliner_model"):
    """Salva il modello trainato"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Salva i pesi del modello
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))
    
    # Salva anche i pesi specifici dei componenti trainati
    torch.save({
        'labels_encoder': core.token_rep_layer.labels_encoder.state_dict(),
        'labels_projection': core.token_rep_layer.labels_projection.state_dict(),
        'epoch': epochs,
        'labels': labels,
        'label2desc': label2desc,
        'label2id': label2id
    }, os.path.join(save_dir, "training_checkpoint.pt"))
    
    print(f"✅ Modello salvato in: {save_dir}")

# ===============================================================
# 0. SETUP
# ===============================================================
random.seed(42)
torch.manual_seed(42)
device = "cpu"

# ===============================================================
# 1. MODELLO E TOKENIZER
# ===============================================================
model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core = model.model  # SpanModel

txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj    = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Congela il text encoder, allena solo label encoder + proiezione
for p in txt_enc.parameters():
    p.requires_grad = False
for p in core.token_rep_layer.labels_encoder.parameters():
    p.requires_grad = True
for p in core.token_rep_layer.labels_projection.parameters():
    p.requires_grad = True

# ===============================================================
# 2. FUNZIONI BASE
# ===============================================================

def get_span_vec_tokens(text, token_span, tokenizer, encoder):
    """
    Converte lo span (s_word, e_word) in subword indices con word_ids()
    e calcola la media dei vettori subword corrispondenti.
    """
    s_word, e_word = token_span
    words = text.split()  # perché 'text' è costruito da ' '.join(tokens)

    # Tokenizzazione con mapping word→subword
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    word_ids = enc.word_ids(batch_index=0)

    # Trova il primo e ultimo sub-token che corrispondono allo span
    sw = next(i for i, w in enumerate(word_ids) if w == s_word)
    ew = max(i for i, w in enumerate(word_ids) if w == e_word)

    with torch.no_grad():
        hidden = encoder(**{k: enc[k] for k in ["input_ids", "attention_mask"]}).last_hidden_state.squeeze(0)

    return hidden[sw:ew+1].mean(dim=0)


def get_label_vecs(labels):
    """Ottiene gli embedding medi per le descrizioni delle label"""
    batch = lbl_tok(labels, return_tensors="pt", padding=True, truncation=True).to(device)
    out = lbl_enc(**batch)
    attn = batch["attention_mask"].float().unsqueeze(-1)
    he_384 = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
    return proj(he_384)

# ===============================================================
# 3. LABELS E TRAIN DATA
# ===============================================================
with open("label2desc.json") as f:
    label2desc = json.load(f)
with open("label2id.json") as f:
    label2id = json.load(f)

labels = list(label2desc.keys())
desc_texts = [label2desc[l] for l in labels]
id2label = {int(v): k for k, v in label2id.items()}

train_df = pd.read_csv("train_data_balanced.csv")
train_data = [(row["text"], eval(row["entity_span"]), int(row["label_id"])) for _, row in train_df.iterrows()]

# ===============================================================
# 🔍 DEBUG: verifica allineamento token/subword
# ===============================================================

def verify_alignment(sample_size=5):
    print("\n=== 🔍 Verifica allineamento token/subword ===")
    sample_rows = random.sample(train_data, sample_size)
    for text, token_span, gold_id in sample_rows:
        s_word, e_word = token_span
        words = text.split()

        # tokenizza parola per parola per creare il mapping subword→word
        enc = txt_tok(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        word_ids = enc.word_ids(batch_index=0)
        subtoks = txt_tok.convert_ids_to_tokens(enc["input_ids"][0])

        # trova i subtoken che appartengono allo span
        sub_span = [i for i, w in enumerate(word_ids) if w is not None and s_word <= w <= e_word]
        sub_tokens = [subtoks[i] for i in sub_span]

        print("\nText:")
        print(text[:150])
        print(f"Word span: ({s_word}, {e_word}) → parole: {words[s_word:e_word+1]}")
        print(f"Subword tokens: {sub_tokens}")
        print(f"Ricostruito: '{txt_tok.convert_tokens_to_string(sub_tokens)}'")
        print(f"Label: {labels[gold_id]}")
    print("\n✅ Se la ricostruzione coincide con l'entità annotata, l'allineamento è corretto.\n")

# Esegui subito la verifica
verify_alignment(sample_size=5)

# ===============================================================
# 4. TRAINING
# ===============================================================
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(
    list(core.token_rep_layer.labels_encoder.parameters()) +
    list(core.token_rep_layer.labels_projection.parameters()),
    lr=1e-4
)

epochs = 6
accum_steps = 16  # ~batch virtuale da 16
model.train()
core.train()

for epoch in range(epochs):
    random.shuffle(train_data)
    total_loss = 0.0

    opt.zero_grad()
    for step, (text, token_span, gold_id) in enumerate(train_data, start=1):
        label_vecs = get_label_vecs(desc_texts)                 # grad attraverso label-encoder OK
        span_vec = get_span_vec_tokens(text, token_span, txt_tok, txt_enc)  # no grad sul text-encoder
        logits = F.normalize(span_vec, dim=-1) @ F.normalize(label_vecs, dim=-1).T
        loss = criterion(logits.unsqueeze(0), torch.tensor([gold_id], device=logits.device))

        (loss / accum_steps).backward()  # scala la loss per accumulation
        total_loss += loss.item()

        if step % accum_steps == 0:
            opt.step()
            opt.zero_grad()

    # flush finale se restano grad non step-ati
    if step % accum_steps != 0:
        opt.step()
        opt.zero_grad()

    print(f"Epoch {epoch+1}/{epochs} | avg_loss={total_loss/len(train_data):.4f}")

save_trained_model(model)
# ===============================================================
# 5. VALUTAZIONE
# ===============================================================
model.eval()
y_true, y_pred = [], []

for text, token_span, gold_id in train_data:
    span_vec = get_span_vec_tokens(text, token_span, txt_tok, txt_enc)
    label_vecs = get_label_vecs(desc_texts)
    logits = F.normalize(span_vec, dim=-1) @ F.normalize(label_vecs, dim=-1).T
    pred_id = logits.argmax().item()
    y_true.append(gold_id)
    y_pred.append(pred_id)

all_label_ids = list(range(len(labels)))
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0, labels=all_label_ids
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="micro", zero_division=0, labels=all_label_ids
)

print("\n=== 📈 Risultati finali ===")
print(f"Macro F1: {f1_macro:.3f} | Micro F1: {f1_micro:.3f}")
print(f"Macro Precision: {prec_macro:.3f} | Macro Recall: {rec_macro:.3f}")
print(f"Micro Precision: {prec_micro:.3f} | Micro Recall: {rec_micro:.3f}")