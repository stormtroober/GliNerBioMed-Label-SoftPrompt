# Report: Analisi Architettura Cross-Encoder GLiNER e Strategia MLP Soft Prompts

## 1. Analisi "Vivisezione" Cross-Encoder GLiNER

A differenza della variante Bi-Encoder (che usa un encoder separato per i label e uno per il testo), il modello `urchade/gliner_small-v2.1` adotta un'architettura **Cross-Encoder (Single-Encoder / Mono-Encoder)**: il testo e i label vengono processati **insieme** nello stesso transformer.

### **Architettura del Modello**

*   **Backbone**: DeBERTa-v2 (o modello transformer singolo), accessibile via `model.token_rep_layer.bert_layer.model`.
*   **Tokenizer**: Il tokenizer del modello wrapper, usato per sia il testo che i label.
*   **Input Format Nativo GLiNER (senza soft prompt)**:
    ```
    [CLS] [ENT] label1 [ENT] label2 ... [ENT] labelN [SEP] token1 token2 ... tokenM [SEP]
    ```
    Tutti i label vengono concatenati prima del testo, separati dal token speciale `[ENT]`.

### **Confronto con il Bi-Encoder**

| Caratteristica              | Bi-Encoder                          | Cross-Encoder (questo script)         |
|-----------------------------|--------------------------------------|---------------------------------------|
| N. di encoder               | 2 (label encoder + text encoder)    | 1 (encoder condiviso)                |
| Formato input label         | `[CLS] label [SEP]` (sequenze separate) | `[ENT] label1 [ENT] label2 ... [SEP] text [SEP]` (concatenato) |
| Interazione label↔testo     | Solo a posteriori (dot product)     | Tramite self-attention interna        |
| Pre-computabilità label     | ✅ Sì (embedding indipendente)       | ❌ No (dipendono dal contesto del testo)|
| Costo inferenza             | Basso (O(n) in #label)              | Alto (O(n × m) per sequenza completa) |

---

## 2. Strategia di Adattamento: MLP Prompt Encoder

Dato che nel Cross-Encoder label e testo condividono lo stesso spazio di input, è possibile inserire i soft prompt come **sequenza di embedding virtuali** nella parte label della sequenza completa. Il `MLPPromptEncoder` sostituisce il modo in cui vengono costruiti gli embedding dei label.

### **Strategia Adottata: Iniezione Diretta come Embedding**

Il modulo `MLPPromptEncoder` **non usa token ID label nativi** per costruire l'input finale al backbone. Invece:

1.  Prende in input gli **ID tokenizzati delle descrizioni** dei label (`desc_input_ids`).
2.  Li processa attraverso un **MLP** (con skip connection e LayerNorm) per trasformarli in representation ricche.
3.  Aplica un **PromptPooler** (in questo caso modalità `conv1d`) per comprimere la sequenza di ciascun label in una sequenza fissa di lunghezza `PROMPT_LEN = 32`.
4.  Restituisce tensori di shape `(NumLabels, PROMPT_LEN, embed_dim)`.

### **Formato Input al Backbone (con soft prompt)**

```
[CLS] [SOFT_PROMPT_label1 × 32 tokens] [SOFT_PROMPT_label2 × 32 tokens] ... [SEP] token1 token2 ... tokenM [SEP]
```

Ovvero, in modo schematico:

```
[CLS]  ←  1 token (embedding CLS del backbone)
[SOFT_PROMPT]  ←  NumLabels × PROMPT_LEN token virtuali (output del MLPPromptEncoder)
[SEP]  ←  1 token separatore
[TEXT TOKENS]  ←  i token del testo originale
[SEP]  ←  1 token di chiusura
```

Tutti i componenti vengono **concatenati a livello di embedding** (`inputs_embeds`) prima di passare all'encoder backbone, bypassando l'embedding lookup nativo.

---

## 3. Dettagli Implementativi del `MLPPromptEncoder`

### **Classe `MLPPromptEncoder`**

```python
class MLPPromptEncoder(nn.Module):
    def __init__(self, original_embeddings, vocab_size, embed_dim,
                 hidden_dim=None, dropout=0.1, prompt_len=None, pooling_mode="adaptive_avg"):
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Copia iniziale del vocab backbone
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),  # hidden_dim = embed_dim * 4 (default)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pooler = PromptPooler(embed_dim, prompt_len, mode=pooling_mode)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)       # Lookup embedding
        x = self.norm(x + self.mlp(x))     # MLP con residual connection
        x = self.pooler(x, attention_mask)  # Pooling a PROMPT_LEN tokens fissi
        return x                            # Shape: (NumLabels, PROMPT_LEN, embed_dim)
```

**Nota**: La embedding table viene inizializzata come **copia** di quella del backbone (`original_embeddings`) ma rimane **addestrabile** come parte del `MLPPromptEncoder`. Tuttavia, nel conteggio dei parametri trainable viene **esclusa** dal totale riportato (è una copia del vocab preesistente, non un parametro genuinamente nuovo).

### **Classe `PromptPooler` — Modalità `conv1d`**

La modalità `conv1d` (usata in questo script) applica due strati Conv1D con GELU, seguiti da `AdaptiveAvgPool1d` per comprimere a `PROMPT_LEN` timestep:

```python
self.conv_layers = nn.Sequential(
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
    nn.GELU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
)
self.adaptive_pool = nn.AdaptiveAvgPool1d(prompt_len)
self.norm = nn.LayerNorm(embed_dim)
```

Flusso: `(B, seq_len, D)` → `mask → transpose → conv → AdaptivePool → transpose → LayerNorm` → `(B, PROMPT_LEN, D)`.

---

## 4. Gestione dell'Input al Backbone

### **Costruzione degli `inputs_embeds`**

La sequenza finale viene assemblata **manualmente** nel training loop:

```python
inputs_embeds = torch.cat([
    cls_embed,           # (B, 1, D)
    batch_soft_prompts,  # (B, NumLabels * PROMPT_LEN, D)
    sep_embed,           # (B, 1, D)
    text_embeds,         # (B, TextLen, D)
    sep_embed            # (B, 1, D)
], dim=1)
```

### **Attention Mask**

La maschera di attenzione tiene conto di tutti i segmenti:

```python
full_mask = torch.cat([
    cls_mask,            # (B, 1) — tutto 1
    prompt_mask,         # (B, NumLabels * PROMPT_LEN) — tutto 1
    sep_mask,            # (B, 1) — tutto 1
    batch["attention_mask"],  # (B, TextLen) — con 0 per padding
    sep_mask             # (B, 1) — tutto 1
], dim=1)
```

Viene poi espansa a `(B, 1, 1, SeqLen)` per passarla direttamente all'encoder DeBERTa (`backbone.encoder(...)`).

### **Estrazione delle Rappresentazioni**

Dall'output del backbone si estraggono **due tipi** di rappresentazioni:

| Rappresentazione | Indici nella sequenza         | Shape risultante        |
|------------------|-------------------------------|-------------------------|
| Text tokens      | `[1 + prompts_len + 1 : text_end]` | `(B, TextLen, D)`  |
| Prompt tokens    | `[1 : 1 + prompts_len]`       | `(B, NumLabels×PLen, D)` |

I prompt token vengono poi **reshapati** a `(B, NumLabels, PROMPT_LEN, D)` e **mediati** lungo `PROMPT_LEN` per ottenere un vettore per label: `(B, NumLabels, D)`.

---

## 5. Calcolo della Similarity e Loss

### **Similarità Coseno + Temperatura**

```python
H_text    = F.normalize(text_reps, dim=-1)     # (B, TextLen, D)
H_prompts = F.normalize(prompt_vectors, dim=-1) # (B, NumLabels, D)

logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / TEMPERATURE
# Shape: (B, TextLen, NumLabels)
```

**Temperatura**: `TEMPERATURE = 0.14792...` (ottimizzata via Optuna), scala le logit prima della softmax.

### **Focal Loss + Class-Balanced Weights**

```python
loss = ce_loss(logits.view(-1, num_labels), batch["labels"].view(-1))
```

*   **Focal Loss** con `gamma = 3.0` per down-weightare i token facili (principalmente `O`).
*   **Class-Balanced weights** (`CB_BETA = 0.9999`) per compensare lo sbilanciamento tra classi.

---

## 6. Vincoli sulla Lunghezza della Sequenza

Dato che testo e prompt devono coesistere nella **stessa sequenza**, la lunghezza massima del testo viene ridotta dinamicamente:

```
MAX_MODEL_LEN (512) - (NumLabels × PROMPT_LEN) - 5 (buffer speciali)  =  MAX_TEXT_LEN
```

Questo è il principale trade-off architetturale rispetto al Bi-Encoder: **più label e/o prompt più lunghi riducono il contesto testuale disponibile**.

---

## 7. Risposta ai Dubbi Specifici

*   **Token `[ENT]` nel Cross-Encoder**: I token `[ENT]` del formato nativo GLiNER vengono **completamente bypassati** in questo script. L'iniezione avviene a livello di `inputs_embeds`, non tramite token ID concatenati. Il backbone quindi non "vede" `[ENT]` ma riceve direttamente gli embedding soft dei label.
*   **Backbone frozen**: Il backbone DeBERTa è completamente frozen (`requires_grad=False`). Solo il `MLPPromptEncoder` (MLP + conv pooler + embedding table copiata) viene addestrato, per un numero molto ridotto di parametri trainable.
*   **Self-attention label↔testo**: A differenza del Bi-Encoder, qui il transformer **può** fare attenzione incrociata tra i token del prompt e i token del testo. Questo è il principale vantaggio del Cross-Encoder: le rappresentazioni dei label diventano contestuali al testo di input.
*   **Pre-computazione dei prompt**: I soft prompt vengono calcolati dal `MLPPromptEncoder` una volta per batch (non per sample). Tuttavia, a differenza del Bi-Encoder, le rappresentazioni finali estratte dal backbone **non sono pre-computabili** perché dipendono dal testo.

---

Questo approccio combina l'efficienza parametrica dei soft prompt (solo il piccolo `MLPPromptEncoder` viene addestrato) con la capacità cross-attention del mono-encoder, permettendo ai label di adattarsi al contesto testuale durante l'inferenza.
