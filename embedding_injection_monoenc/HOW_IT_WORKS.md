# Funzionamento di `train_mono.py` - Embedding Injection Mono-Encoder

Questo documento descrive in dettaglio la pipeline di elaborazione del modello Mono-Encoder, dove le descrizioni delle etichette vengono trasformate in embedding (Soft Prompts) e iniettate direttamente nella sequenza di input del backbone principale.

## Pipeline Dettagliata

### 1. Generazione dei Soft Prompts
Per prima cosa, trasformiamo le descrizioni testuali delle label in embedding.
Questi vengono passati all'interno del `MLPPromptEncoder` che li proietta e poi li "comprime" (pooling) per ottenere una rappresentazione a lunghezza fissa (es. 32 token).

```python
# 1. Genero i Prompt Vettoriali dai testi delle descrizioni
# Input: (NumLabels, MaxLen) -> Output: (NumLabels, 32, 768)
soft_prompts = prompt_encoder(desc_input_ids, attention_mask=desc_attn_mask) 
soft_prompts_flat = soft_prompts.view(-1, embed_dim) # Appiattisco in un "nastro" unico
```

### 2. Costruzione dell'Injection
Concateniamo i vari componenti per formare l'input singolo per il modello. La struttura della sequenza è:
1.  Il token `[CLS]`
2.  I **Soft Prompts** (generati al punto 1)
3.  Un token `[SEP]`
4.  Gli **embedding standard del testo** (estratti dal backbone originale)
5.  Un token finale `[SEP]`

**Risultato:** `[CLS] [LABEL_EMBEDS] [SEP] [TEXT_EMBEDS] [SEP]`

```python
# 2. Ottengo gli embedding del testo (normale)
text_embeds = backbone.embeddings(batch["input_ids"])

# 3. Concatenazione Manuale: [CLS] + [PROMPTS] + [SEP] + [TEXT] + [SEP]
inputs_embeds = torch.cat([
    cls_embed,           # Token di inizio
    batch_soft_prompts,  # <--- I MIEI VETTORI INIETTATI
    sep_embed,           # Separatore
    text_embeds,         # Il testo da analizzare
    sep_embed            # Fine
], dim=1)
```

### 3. Elaborazione dell'Encoder
Passiamo questo vettore concatenato nell'`encoder` del backbone.
Nota: In questa fase il backbone è congelato (frozen), quindi funziona solo come estrattore di feature contestuali.

```python
# 4. Passo tutto al Transformer (Bypassando il layer di lookup parole)
outputs = backbone.encoder(inputs_embeds, attention_mask=full_mask...) 
sequence_output = outputs.last_hidden_state
```

### 4. Separazione dei Risultati
Separiamo l'output dell'encoder per distinguere le rappresentazioni relative al testo da quelle relative ai prompt, che ora sono contestualizzate tra loro grazie all'attenzione del Transformer.

```python
# 5. Estraggo le rappresentazioni post-elaborazione
# Parte Testo: da dopo i prompt e il SEP, fino alla fine
text_reps = sequence_output[:, text_start:text_end, :] 

# Parte Prompt: dall'inizio (salto CLS) per la lunghezza dei prompt
prompt_reps_seq = sequence_output[:, 1:1+prompts_len, :] 
```

### 5. Condensazione delle Etichette
Le rappresentazioni delle etichette (prompt) sono state "arricchite" dal contesto del testo, ma sono ancora sequenze di vettori (es. lunghezza 32). Le comprimiamo facendo una media (Mean Pooling) per ottenere **un solo vettore finale per classe**.

```python
# Rimetto in forma (Batch, NumLabels, 32, Dim)
prompt_reps_reshaped = prompt_reps_seq.view(B, num_labels, prompt_len, embed_dim)

# Faccio la media dei 32 vettori -> 1 vettore per classe
prompt_vectors = prompt_reps_reshaped.mean(dim=2) 
```

### 6. Classificazione Finale
Infine, confrontiamo ogni rappresentazione di parola del testo con ogni etichetta condensata. Utilizziamo il prodotto scalare (similarità) per ottenere i punteggi (logits) su cui viene calcolata la loss.

```python
# 6. Calcolo Similarità (Logits)
logits = torch.bmm(H_text, H_prompts.transpose(1, 2)) / TEMPERATURE
```
