# Baseline Evaluation — Span-Based

## 🔍 Configuration
| Parameter | Value |
|---|---|
| **Model** | `urchade/gliner_small-v2.1` |
| **Encoder** | `mono` |
| **Dataset** | `bc5cdr` |
| **Test file** | `test_dataset_span_mono.json` |
| **Test samples** | `2000` |
| **Batch size** | `8` |
| **Timestamp** | `20260224_191720` |

## 🟢 Evaluation 1 — Label Names (Excluding 'O')
Entity types used: ['chemical', 'disease']

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.7176 | 0.5917 | **0.6453** |
| **Micro** | 0.7229 | 0.6030 | **0.6576** |

| Timing | Value |
|---|---|
| **Total inference time** | `53.54 s` |
| **Avg iterations/sec** | `4.67` |
| **Samples/sec** | `37.36` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
chemical                       | 0.7387   | 0.6941   | 0.7157   | 2236    
disease                        | 0.6966   | 0.4894   | 0.5749   | 1792    
```

## 🔵 Evaluation 2 — Label Descriptions (Excluding 'O')
Entity types used: extended descriptions from label2desc.json
Description truncation: `171` token/label (troncato per mono-encoder)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.8714 | 0.0677 | **0.1246** |
| **Micro** | 0.8793 | 0.0705 | **0.1305** |

| Timing | Value |
|---|---|
| **Total inference time** | `151.84 s` |
| **Avg iterations/sec** | `1.65` |
| **Samples/sec** | `13.17` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
chemical                       | 0.8889   | 0.0930   | 0.1684   | 2236    
disease                        | 0.8539   | 0.0424   | 0.0808   | 1792    
```
