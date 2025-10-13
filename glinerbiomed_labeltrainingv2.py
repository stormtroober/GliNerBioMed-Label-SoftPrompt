# -*- coding: utf-8 -*-
"""
Fine-tuning del label encoder di GLiNER-BioMed (token-level, batch=1)
---------------------------------------------------------------------

Flow:
1️⃣ Carica il dataset JSON (token-level)
2️⃣ Congela il text encoder
3️⃣ Allena label encoder + projection
4️⃣ Calcola similarità token ↔ label (dot product)
5️⃣ Applica CrossEntropyLoss ignorando i subtoken (-100)
"""

import os, json, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from gliner import GLiNER
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time
import numpy as np


# ===============================================================
# ⚙️ SETUP BASE
# ===============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

BATCH_SIZE = 8
EPOCHS = 2
LR = 1e-4
WD = 0.01
OUTDIR = "./gliner_label_ft_toklvl_single"

DATA_JSON = "dataset_masked_balanced_bio.json"
LABEL2ID = "label2id.json"
LABEL2DESC = "label2desc.json"
os.makedirs(OUTDIR, exist_ok=True)

# ===============================================================
# 🧩 DATASET (con allineamento subtoken)
# ===============================================================
class TokenJSONDataset(Dataset):
    def __init__(self, json_path, tokenizer, label2id):
        with open(json_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id

    def align_labels_with_subtokens(self, word_labels, word_ids):
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(self.label2id.get(word_labels[word_id], self.label2id["O"]))
            else:
                aligned_labels.append(self.label2id.get(word_labels[word_id], self.label2id["O"]))
            prev_word_id = word_id
        return aligned_labels

    def __getitem__(self, idx):
        rec = self.records[idx]
        toks = rec["tokens"]
        labs = rec["labels"]

        enc = self.tok(
            toks,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False
        )

        word_ids = enc.word_ids(batch_index=0)
        aligned_labels = self.align_labels_with_subtokens(labs, word_ids)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

    def __len__(self):
        return len(self.records)


# ===============================================================
# 🧠 MODELLI
# ===============================================================
print("📦 Caricamento GLiNER-BioMed...")
model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core = model.model
txt_enc = core.token_rep_layer.bert_layer.model
lbl_enc = core.token_rep_layer.labels_encoder.model
proj = core.token_rep_layer.labels_projection

txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path, add_prefix_space=False)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# Congela text encoder
for p in txt_enc.parameters():
    p.requires_grad = False

# ===============================================================
# 🏷️ LABEL SPACE
# ===============================================================
with open(LABEL2ID) as f:
    label2id = json.load(f)
with open(LABEL2DESC) as f:
    label2desc = json.load(f)

id2label = {int(v): k for k, v in label2id.items()}
labels_text = [label2desc[id2label[i]] for i in range(len(label2id))]
num_labels = len(label2id)

# ===============================================================
# 🧾 DATASET
# ===============================================================
dataset = TokenJSONDataset(DATA_JSON, txt_tok, label2id)
print(f"✅ Dataset caricato: {len(dataset)} frasi")

sample = dataset[0]
tokens = txt_tok.convert_ids_to_tokens(sample["input_ids"])
labels = sample["labels"]

print("TOKENS:", tokens[:40])
print("LABELS:", [list(label2id.keys())[list(label2id.values()).index(x.item())] 
                  if x.item() != -100 else "IGNORE" for x in labels[:40]])

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=txt_tok.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
print("Batch shape:", batch["input_ids"].shape)
print("Example lengths:", [len(x) for x in batch["labels"]])
print("Padding check (last tokens):", batch["labels"][0][-10:])


# ===============================================================
# 🔧 FUNZIONI UTILI
# ===============================================================
def encode_label_text_train(desc_texts):
    """Ottiene embedding medio delle descrizioni label (trainabili)."""
    batch = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = lbl_enc(**batch).last_hidden_state
    attn = batch["attention_mask"].float().unsqueeze(-1)
    pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
    return proj(pooled)  # (C, H)

# ===============================================================
# ⚙️ TRAINING (versione con timing e progress bar)
# ===============================================================
core.train()
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

optim = torch.optim.AdamW(
    list(core.token_rep_layer.labels_encoder.parameters()) +
    list(core.token_rep_layer.labels_projection.parameters()),
    lr=LR,
    weight_decay=WD
)

print(f"\n🚀 Inizio training su {device} | batch_size={BATCH_SIZE} | epochs={EPOCHS}\n")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    step_times = []
    start_epoch = time.time()

    # tqdm progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
    moving_loss = []

    for step, batch in enumerate(pbar, 1):
        step_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()

        # 1️⃣ Encode testo (text encoder congelato)
        with torch.no_grad():
            out = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        H_tok = F.normalize(out.last_hidden_state, dim=-1)

        # 2️⃣ Encode descrizioni label (trainabile)
        H_lbl = F.normalize(encode_label_text_train(labels_text), dim=-1)

        # 3️⃣ Similarità dot product
        logits = torch.matmul(H_tok, H_lbl.T)

        # 4️⃣ Loss
        loss = loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1))
        loss.backward()
        optim.step()

        total_loss += loss.item()
        step_time = time.time() - step_start
        step_times.append(step_time)
        moving_loss.append(loss.item())

        # Aggiorna progress bar ogni step
        mean_loss = np.mean(moving_loss[-10:])  # media mobile ultimi 10
        pbar.set_postfix({
            "loss": f"{mean_loss:.3f}",
            "step_time": f"{step_time:.2f}s"
        })

    epoch_time = time.time() - start_epoch
    avg_loss = total_loss / len(loader)
    print(f"🧩 Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | ⏱️ {epoch_time:.1f}s | ⌛ step medio: {np.mean(step_times):.2f}s")


# ===============================================================
# 💾 SALVATAGGIO
# ===============================================================
torch.save({
    "labels_encoder": core.token_rep_layer.labels_encoder.state_dict(),
    "labels_projection": core.token_rep_layer.labels_projection.state_dict(),
    "label2id": label2id,
    "label2desc": label2desc,
}, os.path.join(OUTDIR, "final_label_encoder.pt"))

print(f"✅ Fine training — modello salvato in: {OUTDIR}/final_label_encoder.pt")
