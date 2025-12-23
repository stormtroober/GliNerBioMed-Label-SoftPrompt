# Report: Analisi Architettura Bi-GLiNER e Strategia Soft Prompts

## 1. Analisi "Vivisezione" Bi-GLiNER

Ho analizzato il comportamento del modello `Ihor/gliner-biomed-bi-small-v1.0` (che usa `sentence-transformers/all-MiniLM-L6-v2` come Label Encoder e `DebertaV2` come Text Encoder) tramite script di debug (`vivisect_bi_gliner.py`).

Ecco i risultati confermati:

### **Label Encoder**
*   **Modello**: BERT-based (`all-MiniLM-L6-v2`).
*   **Input**: Il Label Encoder riceve i label come un **batch di sequenze indipendenti**.
*   **Formato Input**: `[CLS] label_text [SEP]`
    *   Esempio: `[CLS] person [SEP]` (IDs: `[101, 2711, 102]`)
*   **Utilizzo di [ENT]**: **NESSUNO**. Il token `[ENT]` non è presente nel vocabolario standard di questo encoder e non viene utilizzato per concatenare i label. I label NON vengono passati in un'unica stringa `[ENT] label1 [ENT] label2`.

### **Text Encoder**
*   **Modello**: DeBERTa-v2.
*   **Input**: Riceve solo il testo di input.
*   **Formato Input**: `[CLS] input_sentence [SEP]` (Token IDs specifici di DeBERTa).
*   **Iniezione Label**: I label **NON sono presenti** nell'input del text encoder.

### **Confronto con l'Ipotesi Utente**
L'ipotesi che l'input segua il formato `[ENT] ent_1 [ENT] ent_2 ... [SEP] input_sentence` è valida per l'architettura **GLiNER Standard (Single/Cross Encoder)**, ma **NON si applica** a questa architettura **Bi-Encoder**.
Nel Bi-Encoder, i due flussi (Testo e Label) sono separati e si incontrano solo al calcolo della similarità (prodotto scalare degli embedding risultanti).

## 2. Strategia di Adattamento Soft Prompts

Dato che il Bi-Encoder processa ogni label singolarmente, non possiamo usare il formato concatenato `[ENT] sp1 [ENT] sp2`. Dobbiamo invece iniettare il soft prompt all'interno della sequenza di *ogni singolo label*.

### **Strategia Proposta**
Inserire i vettori del Soft Prompt (apprendibili) direttamente all'inizio della sequenza di input del **Label Encoder**, subito dopo il token `[CLS]`.

**Formato Input Label Target:**
`[CLS] [SOFT_PROMPT] label_text [SEP]`

Dove `[SOFT_PROMPT]` è una sequenza di $N$ vettori (non token ID, ma embedding diretti) che vengono appresi durante il training.

### **Dettagli Implementativi**
1.  **Label Encoder Hook / Override**: È necessario intercettare l'input del Label Encoder (`input_embeds`).
2.  **Sostituzione Embedding**:
    *   Convertire `[CLS] label [SEP]` in embedding.
    *   Inserire i vettori del soft prompt tra l'embedding di `[CLS]` e l'embedding del label.
    *   Alternativa più semplice: Concatenare il soft prompt embedding ai word embedding dei label.
3.  **Gestione [SEP] extra**: Dato che non stiamo concatenando più label, non è necessario gestire separatori extra tra label diversi, poiché vivono in batch separati.

## 3. Risposta ai Dubbi Specifici
*   **Token [ENT] (128002) e [SEP] (128003)**: Questi ID appartengono verosimilmente al tokenizer di un modello diverso (es. GLiNER Base basato su LLaMA o DeBERTa custom). Il modello attuale `all-MiniLM-L6-v2` usa il vocabolario BERT standard (30k token) dove `[SEP]` è 102. Non dobbiamo preoccuparci degli ID 128002/128003 per *questo* specifico Bi-Encoder, a meno di non voler cambiare tokenizer.
*   **Formattazione Soft Prompt**: Non diventerà `[ENT] soft_prompt_1 [ENT] ...`, ma sarà un'iniezione locale per ogni label processato.

Questo approccio mantiene la coerenza con l'architettura Bi-Encoder (efficienza e pre-computazione) pur permettendo al modello di apprendere una rappresentazione "soft" delle classi.
