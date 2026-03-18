# GliNerBioMed — Label & Soft Prompt Tuning

> **Master's Thesis** · Alma Mater Studiorum – University of Bologna  
> *Enhancing Entity Representations through Soft Prompt Tuning: An Efficient Approach to Biomedical Named Entity Recognition*  
> Author: **Alessandro Becci** · Supervisor: Prof. Gianluca Moro

---

## Overview

This repository contains the full experimental codebase developed for a comparative study of **Parameter-Efficient Fine-Tuning (PEFT)** strategies applied to [GLiNER](https://github.com/urchade/GLiNER) for **Biomedical Named Entity Recognition (BioNER)**.

GLiNER is a zero-shot, span-based NER model that reformulates entity recognition as a *span–label matching* problem. While powerful for general-domain use, its out-of-the-box performance degrades in biomedicine — where dense technical terminology and complex nomenclature demand domain-specific adaptation. Rather than resorting to expensive full fine-tuning, this work investigates how targeted, parameter-efficient approaches can close that gap.

Five distinct adaptation strategies are benchmarked on **JNLPBA** (5 entity types: protein, DNA, RNA, cell type, cell line) and **BC5CDR** (2 entity types: chemical, disease).

---

## Approaches

### 1. Label Descriptions as Semantic Anchors (LDSA)
`hardprompting/`

A description-driven **bi-encoder baseline**. The text encoder is frozen entirely; only the label encoder and a linear projection layer are trained. Entity types are grounded via rich natural-language descriptions (e.g., *"Chemical: a substance with distinct molecular composition"*) rather than bare labels, injecting domain semantics directly into the label representation space.

- Trainable components: label encoder + projection layer (~10% of parameters)
- Loss: Cross-entropy with inverse-frequency class weighting
- Hypothesis: adapting the label space alone is sufficient for domain specialization

### 2. Soft Prompting — Base Approach
`softprompting/`

Replaces the entire label encoder with a lightweight **trainable embedding table** (one vector per entity type). Entity descriptions initialize the table via a one-shot encoder pass; the encoder is then discarded. Gradients update the embeddings directly through the shared embedding space — no linguistic constraint, no architecture overhead.

Three training variants were developed iteratively:
- `v1`: Standard cross-entropy + inverse-frequency weighting
- `focal`: Focal Loss to down-weight easy background tokens
- `focal_cb` *(selected)*: Class-Balanced Focal Loss with split learning rates for the embedding table and projection layer

### 3. Soft Prompting — Embedding Injection
`embedding_injection_monoenc/` · `embedding_injection_bienc/`

Learnable prompts are generated from entity descriptions via a trainable **MLPPromptEncoder** (residual MLP + LayerNorm + PromptPooler) and injected directly into the Transformer's input sequence, enabling contextual refinement through self-attention.

Two architectural variants:

| Variant | Injection point | Interaction |
|---|---|---|
| **Cross-Encoder** (`SP-Cross`) | Unified input sequence: `[CLS] + Prompts + [SEP] + Text + [SEP]` | Full bidirectional attention between prompts and text |
| **Bi-Encoder** (`SP-Bi`) | Label branch only: `[CLS] + Prompts + [SEP]` per entity type | Independent encoding; prompts attend only to themselves |

The Cross-Encoder achieves richer cross-modal interaction but faces a sequence-length trade-off (prompt tokens consume capacity from the 512-token budget). The Bi-Encoder avoids this constraint and allows label representations to be pre-computed.

Each variant includes separate Optuna-driven hyperparameter searches for architecture (pooling strategy, prompt length) and training (learning rate, focal loss γ, β).

### 4. Custom GLiNER Extension (SP-GLiNER)
`bienc_embedding_glinerev/`

The most expressive PEFT approach. Instead of adding new tokens, a lightweight **PromptEncoder** (single linear layer + residual connection + LayerNorm) intercepts the label encoder's embedding layer and transforms label-token embeddings *in-place*, before the frozen Transformer backbone processes them. Special tokens (`[CLS]`, `[SEP]`) are excluded via a binary position mask, preserving the backbone's attention patterns.

What makes it unique:
- No synthetic tokens are added — the sequence structure remains natural
- Downstream GLiNER components (`rnn`, `span_rep_layer`, `prompt_rep_layer`) are **jointly fine-tuned** alongside the injection module
- Three-tier learning rate schedule: PromptEncoder (highest) → downstream layers (moderate) → frozen backbone

This approach trains only **~13% of total model parameters**.

### 5. Baseline Fine-tuning (Full-FT)
`finetune/`

End-to-end fine-tuning of all model parameters using GLiNER's official training API, for both Cross-Encoder (`gliner_small-v2.1`) and Bi-Encoder (`gliner-biomed-bi-small-v1.0`) configurations. Serves as the **performance upper bound** against which all PEFT methods are evaluated.

---

## Results

### Span-level F1 (Exact Match)

| Approach | JNLPBA Macro F1 | BC5CDR Macro F1 | Trainable Params |
|---|---|---|---|
| Zero-Shot Baseline | ~0.38 | ~0.61 | 0% |
| LDSA | ~0.52 | ~0.73 | ~10% |
| SP-Base | ~0.54 | ~0.75 | ~5% |
| SP-Cross | ~0.60 | ~0.79 | ~3% |
| SP-Bi | ~0.62 | ~0.80 | ~3% |
| **SP-GLiNER (Custom Extension)** | **0.7016** | **0.8526** | **~13%** |
| Full Fine-tuning (reference) | 0.7164 | 0.8833 | 100% |

> **Key finding:** The Custom GLiNER Extension achieves performance within ~1.5 pp of full fine-tuning on both benchmarks while updating only 13% of model parameters.

---

## Dataset Generation
`dataset_generation/`

A unified pipeline transforms raw BIO-tagged datasets (JNLPBA, BC5CDR from HuggingFace) into training-ready formats for both encoder architectures. It produces:

- **Token-level** datasets (`dataset_tknlvl_{cross|bi}.json`) for LDSA and Soft Prompting approaches
- **Span-level** datasets (`dataset_span_{cross|bi}.json`) for GLiNER-native span scoring
- **Metadata** files: `label2desc.json` (entity descriptions from Pile-NER-biomed), `label2id.json`

Entity descriptions are sourced from the `disi-unibo-nlp/Pile-NER-biomed-descriptions` dataset and used as semantic grounding for Embedding Injection and LDSA.

---

## Hyperparameter Optimization

All approaches use **Optuna** (TPE sampler + MedianPruner) for automated hyperparameter search. Results and importance plots are saved per approach and dataset. A helper script generates all Optuna visualizations:

```bash
python generate_optuna_plots.py --config example_boolean_params.json
bash example_generate_plots.sh
```

---

## Project Structure

```
.
├── dataset_generation/         # BIO→span/token pipeline for JNLPBA & BC5CDR
├── hardprompting/              # LDSA: label encoder fine-tuning
├── softprompting/              # SP-Base: trainable embedding table (v1, focal, focal+CB)
├── embedding_injection_monoenc/# SP-Cross: MLP prompt injection into Cross-Encoder
├── embedding_injection_bienc/  # SP-Bi: MLP prompt injection into Bi-Encoder
├── bienc_embedding_glinerev/   # SP-GLiNER: Custom GLiNER Extension (in-place injection)
├── finetune/                   # Baseline full fine-tuning (bi + cross encoder)
├── baselinetests/              # Zero-shot baseline evaluation
├── generate_optuna_plots.py    # Unified Optuna visualization script
└── requirements.txt
```

Each approach folder follows this convention:
- `train_*.py` — training script(s), including Optuna variants
- `test_*.py` — evaluation script (loads checkpoint from `savings/`, writes to `test_results/`)
- `savings/` — trained `.pt` checkpoints (timestamped)
- `optuna*/` — Optuna study artifacts and plots

---

## Setup

```bash
# Create and activate virtual environment
uv venv venvgliner
source venvgliner/bin/activate

# Install PyTorch (CPU build)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
uv pip install -r requirements.txt
```

**Key dependencies:** `gliner==0.2.22`, `transformers==4.51.0`, `optuna` (installed via gliner), `torch`, `scikit-learn`, `pandas`

---

## Conclusions

The **Custom GLiNER Extension** demonstrates that PEFT can be a viable alternative to full fine-tuning in specialized NER, provided the adaptation targets the right architectural components. By operating directly in the embedding space at the token level — rather than adding synthetic context — and jointly fine-tuning downstream span-scoring layers, the approach achieves compelling efficiency/performance trade-offs.

Across all PEFT methods, a clear progression emerges: richer adaptation (from frozen label encoder → soft embedding table → injected prompts → in-place embedding transformation) consistently improves performance. However, only the Custom GLiNER Extension, by bridging prompt-based adaptation with native span-level fine-tuning, manages to come close to the full fine-tuning reference.

---