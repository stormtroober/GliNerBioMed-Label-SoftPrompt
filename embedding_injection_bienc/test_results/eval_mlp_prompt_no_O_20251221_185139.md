# Risultati Test (NO 'O' CLASS) - MLP Prompt Encoder

> ℹ️ **NOTA**: Le metriche Globali e il Report di Classificazione ESCLUDONO la classe 'O' (Non-classe).

**Checkpoint:** `mlp_focal_cbclass_val-20251221_184944.pt`

**Dataset:** `../dataset/test_dataset_tokenlevel.json`

**Data:** 2025-12-21 18:51:39

---

## 🎯 CONFIGURAZIONE PROMPT TUNING

| Parametro | Valore |
|:----------|:-------|
| **🔧 Prompt Length** | **32** |
| **🔧 Pooling Mode** | **conv1d** |
| **📝 Descrizione** | Conv1D + AdaptiveAvgPool |
| **Projection Trained** | ✅ Sì |

---

## ⚙️ Parametri di Training Salvati

| Parametro | Valore |
|:----------|:-------|
| **prompt_len** | **32** |
| **pooling_mode** | **conv1d** |
| batch_size | 128 |
| epochs | 1 |
| dataset_size | 5000 |
| lr_mlp | 0.002 |
| lr_proj | 0.002 |
| temperature | 0.011641058260782156 |
| gamma_focal_loss | 5.0 |
| cb_beta | 0.9999 |
| dropout_rate | 0.1 |
| early_stopping_patience | 5 |
| grad_clip | 1.0 |
| random_seed | 42 |
| train_size | 4500 |
| val_size | 500 |
| validation_ratio | 0.1 |
| validation_split | True |
| warmup_ratio | 0.10603059187238079 |
| warmup_steps | 3 |
| weight_decay | 0.01 |
| weight_strategy | ClassBalanced |

## 📈 Metriche Globali (ESCLUSO 'O')

### Riassunto Performance
| Average Type | Precision | Recall | F1-Score |
|:-------------|----------:|-------:|---------:|
| **Macro**    | 0.0733 | 0.1660 | **0.1017** |
| **Micro**    | 0.3661 | 0.4414 | **0.4002** |
| Weighted     | 0.1949 | 0.4414 | 0.2704 |

**Token Totali Valutati**: 6,523

## 📊 Metriche per Classe (Tutte incluse)

| Classe | Precision | Recall | F1-Score | Support | Predicted |
|:-------|----------:|-------:|---------:|--------:|----------:|
| cell type | 0.0000 | 0.0000 | 0.0000 | 417 | 0 |
| rna | 0.0000 | 0.0000 | 0.0000 | 18 | 2 |
| cell line | 0.0000 | 0.0000 | 0.0000 | 80 | 0 |
| dna | 0.0000 | 0.0000 | 0.0000 | 252 | 0 |
| protein | 0.3664 | 0.8301 | 0.5084 | 871 | 1973 |
| **O** (Esclusa dal Macro/Micro) | 0.9409 | 0.8759 | 0.9072 | 4885 | 4548 |
| **TOTAL** | - | - | - | 6523 | 6523 |

## 📋 Classification Report (No 'O')

```
              precision    recall  f1-score   support

   cell type     0.0000    0.0000    0.0000       417
         rna     0.0000    0.0000    0.0000        18
   cell line     0.0000    0.0000    0.0000        80
         dna     0.0000    0.0000    0.0000       252
     protein     0.3664    0.8301    0.5084       871

   micro avg     0.3661    0.4414    0.4002      1638
   macro avg     0.0733    0.1660    0.1017      1638
weighted avg     0.1949    0.4414    0.2704      1638

```

## 🔢 Distribuzione Predizioni vs Ground Truth

| Classe | Predette | Vere | Differenza | % Coverage |
|:-------|:--------:|:----:|:----------:|:----------:|
| O | 4548 | 4885 | -337 | 93.1% |
| protein | 1973 | 871 | +1102 | 226.5% |
| rna | 2 | 18 | -16 | 11.1% |

## 🔀 Top Confusioni (Errori più frequenti)

| Vera Classe | Predetta Come | Count |
|:------------|:--------------|------:|
| O | protein | 605 |
| cell type | protein | 341 |
| dna | protein | 218 |
| protein | O | 148 |
| cell type | O | 75 |
| cell line | protein | 70 |
| dna | O | 34 |
| rna | protein | 16 |
| cell line | O | 10 |
| rna | O | 2 |
| cell type | rna | 1 |
| O | rna | 1 |
