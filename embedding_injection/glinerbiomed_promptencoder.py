# -*- coding: utf-8 -*-
"""
Training Prompt Encoder esterno per Bi-GLiNER.
Il backbone (Text Encoder + Label Encoder + Projection) è congelato (Blackbox).
Viene addestrato solo il Prompt Encoder che genera input ottimizzati per il Label Encoder.
"""

import json
import torch
import torch.nn.functional as F
import time
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gliner import GLiNER
from tqdm import tqdm

# ==========================================================
# 🔧 CONFIGURAZIONE
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-4  # Solitamente il prompt tuning richiede LR leggermente più alto
WEIGHT_DECAY = 0.01
TEMPERATURE = 1.0
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42

DATASET_PATH = "../dataset/dataset_tokenlevel_balanced.json"
LABEL2DESC_PATH = "../label2desc.json"
LABEL2ID_PATH = "../label2id.json"
MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# 1️⃣ DEFINIZIONE DEL PROMPT ENCODER
# ==========================================================
class PromptEncoder(nn.Module):
    """
    Modulo esterno che sostituisce il layer di embedding iniziale del Label Encoder.
    Impara rappresentazioni 'soft' dei token delle descrizioni.
    """
    def __init__(self, original_embeddings, vocab_size, embed_dim):
        super().__init__()
        # Inizializziamo con i pesi originali del modello per partire da una base sensata
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(original_embeddings.weight)
        
    def forward(self, input_ids):
        # input_ids: [num_labels, seq_len]
        # output: [num_labels, seq_len, embed_dim] (Soft Embeddings)
        return self.embedding(input_ids)

# ==========================================================
# 2️⃣ PREPARAZIONE DEL MODELLO (BLACKBOX STRATEGY)
# ==========================================================
print("📦 Caricamento modello base GLiNER-BioMed...")
model = GLiNER.from_pretrained(MODEL_NAME)
core = model.model

# Estrazione componenti
txt_enc = core.token_rep_layer.bert_layer.model     # DeBERTa
lbl_enc = core.token_rep_layer.labels_encoder.model # MiniLM (BERT architecture)
proj = core.token_rep_layer.labels_projection       # Linear 384->768

# Tokenizers
txt_tok = AutoTokenizer.from_pretrained(txt_enc.config._name_or_path)
lbl_tok = AutoTokenizer.from_pretrained(lbl_enc.config._name_or_path)

# 🔒 FREEZE TOTALE DEL BACKBONE (Ste's Blackbox Rule)
for p in txt_enc.parameters(): p.requires_grad = False
for p in lbl_enc.parameters(): p.requires_grad = False
for p in proj.parameters():    p.requires_grad = False  # Anche la proiezione è congelata!

print("✅ Text Encoder, Label Encoder e Proiezione congelati.")

# Creazione del Prompt Encoder esterno
# Prendiamo la dimensione dagli embedding originali del Label Encoder (MiniLM)
original_word_embeddings = lbl_enc.embeddings.word_embeddings
vocab_size = original_word_embeddings.num_embeddings
embed_dim = original_word_embeddings.embedding_dim

prompt_encoder = PromptEncoder(original_word_embeddings, vocab_size, embed_dim).to(DEVICE)

print(f"✨ Prompt Encoder creato (Vocab: {vocab_size}, Dim: {embed_dim}) - UNICO TRAINABLE")

# ==========================================================
# 3️⃣ PREPARAZIONE DESCRIZIONI (INPUT PER PROMPT ENCODER)
# ==========================================================
with open(LABEL2DESC_PATH) as f: label2desc = json.load(f)
with open(LABEL2ID_PATH) as f: label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
label_names = [id2label[i] for i in range(num_labels)]

print(f"✅ Caricate {num_labels} label.")

# Tokenizziamo le descrizioni UNA SOLA VOLTA
# Questo sarà l'input fisso per il nostro Prompt Encoder
print("⚙️  Tokenizzazione descrizioni...")
desc_texts = [label2desc[name] for name in label_names]

# Tokenizzazione con padding batch
desc_inputs = lbl_tok(desc_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
desc_input_ids = desc_inputs["input_ids"]       # Input per il Prompt Encoder
desc_attn_mask = desc_inputs["attention_mask"]  # Input per il Label Encoder (per mascherare il padding)

print(f"✅ Descrizioni pronte: {desc_input_ids.shape}")

# ==========================================================
# 4️⃣ DATASET E UTILS
# ==========================================================
class TokenJsonDataset(Dataset):
    def __init__(self, path_json, tokenizer, label2id):
        with open(path_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tok = tokenizer
        self.label2id = label2id

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        labels = rec["labels"]
        input_ids = self.tok.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

def collate_batch(batch, pad_id, ignore_index=-100):
    maxlen = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), ignore_index, dtype=torch.long)
    
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attention_mask"]
        labels[i, :L] = ex["labels"]
    
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

def compute_class_weights(data_path, label2id):
    with open(data_path, "r") as f: data = json.load(f)
    counts = torch.zeros(len(label2id))
    total = 0
    for record in data:
        for label in record["labels"]:
            if label != -100:
                counts[label] += 1
                total += 1
    weights = total / (len(label2id) * counts.clamp(min=1))
    return weights

# ==========================================================
# 5️⃣ FUNZIONE DI TRAINING CORE
# ==========================================================
def get_label_embeddings_from_prompt(prompt_enc, frozen_lbl_enc, frozen_proj, input_ids, attention_mask):
    """
    Pipeline custom: 
    Input IDs -> Prompt Encoder -> Soft Embeds -> Frozen Label Encoder (via inputs_embeds) -> Frozen Proj
    """
    # 1. Genera Soft Embeddings (l'unica parte che calcola gradienti su prompt_enc)
    soft_embeds = prompt_enc(input_ids) # [num_labels, seq_len, 384]
    
    # 2. Inietta nel Label Encoder (MiniLM)
    # Usiamo inputs_embeds per saltare il word embedding interno di MiniLM
    # NOTA: MiniLM aggiungerà internamente position embeddings e token_type embeddings ai nostri soft_embeds.
    outputs = frozen_lbl_enc(inputs_embeds=soft_embeds, attention_mask=attention_mask)
    
    # 3. Mean Pooling (standard per Sentence Transformers)
    # O CLS pooling, a seconda di come è stato trainato MiniLM. Di solito Mean Pooling è meglio per sentence-transformers.
    # Qui replichiamo la logica del codice precedente: Weighted Mean Pooling
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask # [num_labels, 384]
    
    # 4. Proiezione finale (Congelata)
    projected = frozen_proj(pooled)    # [num_labels, 768]
    
    return projected

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# ==========================================================
# 6️⃣ TRAINING LOOP
# ==========================================================
print("\n📚 Caricamento dataset...")
ds = TokenJsonDataset(DATASET_PATH, txt_tok, label2id)
class_weights = compute_class_weights(DATASET_PATH, label2id).to(DEVICE)
ce_loss = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

train_loader = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: collate_batch(b, pad_id=txt_tok.pad_token_id)
)

# Optimizer: SOLO per Prompt Encoder
optimizer = optim.AdamW(prompt_encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

print("\n🚀 Inizio training (Prompt Tuning)...")

txt_enc.eval()
lbl_enc.eval()
proj.eval()
prompt_encoder.train() # Solo questo è in train mode

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    prompt_encoder.train()
    
    # Calcoliamo le Label Representations una volta per batch? 
    # No, dobbiamo ricalcolarle ogni step perché il Prompt Encoder cambia e dobbiamo propagare i gradienti attraverso di esse.
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in progress_bar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        
        # --- 1. Text Side (Frozen) ---
        with torch.no_grad():
            out_txt = txt_enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            H_text = F.normalize(out_txt.last_hidden_state, dim=-1) # [B, Seq, 768]
            
        # --- 2. Label Side (Trainable Prompt -> Frozen Label Enc -> Frozen Proj) ---
        # Passiamo le descrizioni fisse attraverso il Prompt Encoder che stiamo allenando
        label_embeds = get_label_embeddings_from_prompt(
            prompt_encoder, lbl_enc, proj, 
            desc_input_ids, desc_attn_mask
        )
        label_matrix = F.normalize(label_embeds, dim=-1) # [Num_Labels, 768]
        
        # --- 3. Similarity & Loss ---
        # Calcolo similarità (Bi-Encoder logic)
        logits = torch.matmul(H_text, label_matrix.T) / TEMPERATURE # [B, Seq, Num_Labels]
        
        loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Salviamo solo il prompt encoder
        os.makedirs("savings", exist_ok=True)
        torch.save(prompt_encoder.state_dict(), "savings/best_prompt_encoder.pt")
    
    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print("🛑 Early Stopping")
        break

# ==========================================================
# 💾 EXPORT FINALE
# ==========================================================
print("\n✅ Training completato.")
print("Il modello Bi-GLiNER originale è rimasto intatto.")
print("Puoi caricare 'best_prompt_encoder.pt' e usarlo per iniettare embeddings ottimizzati nel Label Encoder.")