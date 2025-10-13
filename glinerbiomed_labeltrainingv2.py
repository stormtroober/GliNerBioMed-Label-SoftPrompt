# step2_gliner_check.py
import torch
from gliner import GLiNER
import json
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from gliner import GLiNER
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Carica il modello completo (GLiNER BioMed)
print("📥 Caricamento GLiNER BioMed...")
model = GLiNER.from_pretrained("Ihor/gliner-biomed-bi-small-v1.0")
core = model.model  # SpanModel interno

# Estrarre i componenti
txt_enc = core.token_rep_layer.bert_layer.model          # encoder del testo
lbl_enc = core.token_rep_layer.labels_encoder.model      # encoder delle descrizioni
proj    = core.token_rep_layer.labels_projection         # proiezione label -> dim testo


print(txt_enc.config._name_or_path)
print(lbl_enc.config._name_or_path)
print("\n✅ Estratti encoder GLiNER:")
print(" - Text encoder:", type(txt_enc))
print(" - Label encoder:", type(lbl_enc))
print(" - Projection:", proj)

# 2️⃣ Carica mapping e descrizioni
LABEL2DESC = "label2desc.json"
label2desc = json.loads(Path(LABEL2DESC).read_text())
if "O" not in label2desc:
    label2desc["O"] = "outside / non-entity token"
labels = list(label2desc.keys())
desc_texts = [label2desc[l] for l in labels]
print(f"\n📚 Label set ({len(labels)}):", labels)

from transformers import AutoTokenizer
import torch, json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Ricava i tokenizer corretti dagli encoder
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config.name_or_path)
lab_tok = AutoTokenizer.from_pretrained(lbl_enc.config.name_or_path)

# 2) Encoda le descrizioni con il label encoder (+ proiezione)
enc_lab = lab_tok(desc_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    Hlab = lbl_enc(**enc_lab).last_hidden_state[:, 0, :]   # [K, 384]
    Hlab_proj = proj(Hlab)                                  # [K, 768]

print(f"[OK] Hlab raw {tuple(Hlab.shape)} → proj {tuple(Hlab_proj.shape)}")

# 3) Test rapido sul testo GREZZO (non usare il dataset token-level qui)
text = "T cell priming enhances IL-4 gene expression by increasing nuclear factor of activated T cells."
enc_txt = txt_tok(text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    Htxt = txt_enc(**enc_txt).last_hidden_state             # [1, L, 768]
print(f"[OK] Htxt {tuple(Htxt.shape)}")

# 4) Similarità token–label
Htxt_n = torch.nn.functional.normalize(Htxt, dim=-1)
Hlab_n = torch.nn.functional.normalize(Hlab_proj, dim=-1)
S = torch.einsum("bld,kd->blk", Htxt_n, Hlab_n)             # [1, L, K]

print(f"[OK] S {tuple(S.shape)}  mean={S.mean().item():.3f}±{S.std().item():.3f}")
pred = S[0].argmax(-1)
print("Primi 10 token e pred:")
for t, p in zip(txt_tok.convert_ids_to_tokens(enc_txt["input_ids"][0])[:10], pred[:10].tolist()):
    print(f"{t:15s} → {labels[p]}")


for p in txt_enc.parameters():
    p.requires_grad = False  # FREEZE
print("✅ Text encoder frozen")

train_params = list(lbl_enc.parameters()) + list(proj.parameters())
print(f"Trainable params: {sum(p.numel() for p in train_params)/1e6:.2f}M")

# === 2️⃣ Tokenizer e label mapping ===
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
LABEL2ID = "label2id.json"
label2id = json.loads(Path(LABEL2ID).read_text())
K = len(label2id)
id2label = {v: k for k, v in label2id.items()}

# === 3️⃣ Dataset loader ===
DATA_JSON = "dataset_masked_balanced_bio.json"
recs = json.loads(Path(DATA_JSON).read_text())

def collate_fn(batch):
    maxlen = max(len(r["tokens"]) for r in batch)
    pad_id = txt_tok.pad_token_id or 0
    input_ids, attn, labels = [], [], []
    for r in batch:
        ids = txt_tok.convert_tokens_to_ids(r["tokens"])
        ids += [pad_id]*(maxlen-len(ids))
        mask = [1]*len(r["tokens"]) + [0]*(maxlen-len(r["tokens"]))
        lab  = [(-100 if l=="IGNORE" else label2id.get(l,label2id["O"])) for l in r["labels"]]
        lab += [-100]*(maxlen-len(lab))
        input_ids.append(ids); attn.append(mask); labels.append(lab)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

loader = DataLoader(recs, batch_size=4, shuffle=True, collate_fn=collate_fn)

# === 4️⃣ Training loop ===
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
opt = torch.optim.AdamW(train_params, lr=2e-5, weight_decay=0.01)

for epoch in range(3):
    lbl_enc.train(); proj.train()
    total_loss, steps = 0.0, 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(DEVICE) for k,v in batch.items()}

        with torch.no_grad():
            Htxt = txt_enc(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).last_hidden_state  # [B,L,768]

        # Forward label encoder + projection
        descs = list(label2id.keys())
        enc_lab = core.token_rep_layer.labels_encoder.tokenizer(
            descs, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        Hlab = lbl_enc(**enc_lab).last_hidden_state[:,0,:]  # [K,384]
        Hlab_proj = proj(Hlab)                              # [K,768]

        # Similarità
        Htxt_n = torch.nn.functional.normalize(Htxt, dim=-1)
        Hlab_n = torch.nn.functional.normalize(Hlab_proj, dim=-1)
        sims = torch.einsum("bld,kd->blk", Htxt_n, Hlab_n)

        loss = loss_fn(sims.view(-1,K), batch["labels"].view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        steps += 1
        if steps % 20 == 0:
            print(f"  step {steps} loss={total_loss/steps:.4f}")

    print(f"Epoch {epoch+1} avg_loss={total_loss/steps:.4f}")

torch.save({
    "lbl_enc": lbl_enc.state_dict(),
    "proj": proj.state_dict()
}, "trained_label_encoder.pt")

print("✅ Training completo e pesi salvati in trained_label_encoder.pt")