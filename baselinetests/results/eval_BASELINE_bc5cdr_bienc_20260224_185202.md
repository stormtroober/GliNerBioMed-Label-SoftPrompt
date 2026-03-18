# Baseline Evaluation — Span-Based

## 🔍 Configuration
| Parameter | Value |
|---|---|
| **Model** | `Ihor/gliner-biomed-bi-small-v1.0` |
| **Encoder** | `bi` |
| **Dataset** | `bc5cdr` |
| **Test file** | `test_dataset_span_bi.json` |
| **Test samples** | `2000` |
| **Batch size** | `8` |
| **Timestamp** | `20260224_185202` |

## 🟢 Evaluation 1 — Label Names (Excluding 'O')
Entity types used: ['chemical', 'disease']

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.7246 | 0.6531 | **0.6859** |
| **Micro** | 0.7347 | 0.6641 | **0.6976** |

| Timing | Value |
|---|---|
| **Total inference time** | `66.94 s` |
| **Avg iterations/sec** | `3.73` |
| **Samples/sec** | `29.88` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
chemical                       | 0.7782   | 0.7531   | 0.7655   | 2236    
disease                        | 0.6710   | 0.5530   | 0.6063   | 1792    
```

## 🔵 Evaluation 2 — Label Descriptions (Excluding 'O')
Entity types used: extended descriptions from label2desc.json

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.4067 | 0.0605 | **0.0968** |
| **Micro** | 0.4790 | 0.0539 | **0.0969** |

| Timing | Value |
|---|---|
| **Total inference time** | `62.73 s` |
| **Avg iterations/sec** | `3.99` |
| **Samples/sec** | `31.88` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
chemical                       | 0.3333   | 0.0004   | 0.0009   | 2236    
disease                        | 0.4800   | 0.1205   | 0.1927   | 1792    
```
