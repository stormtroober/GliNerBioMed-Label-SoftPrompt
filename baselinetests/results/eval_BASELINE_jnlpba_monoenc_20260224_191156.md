# Baseline Evaluation — Span-Based

## 🔍 Configuration
| Parameter | Value |
|---|---|
| **Model** | `urchade/gliner_small-v2.1` |
| **Encoder** | `mono` |
| **Dataset** | `jnlpba` |
| **Test file** | `test_dataset_span_mono.json` |
| **Test samples** | `2000` |
| **Batch size** | `8` |
| **Timestamp** | `20260224_191156` |

## 🟢 Evaluation 1 — Label Names (Excluding 'O')
Entity types used: ['cell line', 'cell type', 'dna', 'protein', 'rna']

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.3501 | 0.2823 | **0.2726** |
| **Micro** | 0.5603 | 0.4397 | **0.4927** |

| Timing | Value |
|---|---|
| **Total inference time** | `55.97 s` |
| **Avg iterations/sec** | `4.47` |
| **Samples/sec** | `35.73` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
cell line                      | 0.2596   | 0.5240   | 0.3472   | 271     
cell type                      | 0.7260   | 0.2915   | 0.4160   | 1036    
dna                            | 0.1667   | 0.0018   | 0.0036   | 546     
protein                        | 0.5985   | 0.5943   | 0.5964   | 2581    
rna                            | 0.0000   | 0.0000   | 0.0000   | 67      
```

## 🔵 Evaluation 2 — Label Descriptions (Excluding 'O')
Entity types used: extended descriptions from label2desc.json
Description truncation: `67` token/label (troncato per mono-encoder)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.1900 | 0.0012 | **0.0024** |
| **Micro** | 0.5185 | 0.0031 | **0.0062** |

| Timing | Value |
|---|---|
| **Total inference time** | `379.60 s` |
| **Avg iterations/sec** | `0.66` |
| **Samples/sec** | `5.27` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
cell line                      | 0.0000   | 0.0000   | 0.0000   | 271     
cell type                      | 0.0833   | 0.0010   | 0.0019   | 1036    
dna                            | 0.0000   | 0.0000   | 0.0000   | 546     
protein                        | 0.8667   | 0.0050   | 0.0100   | 2581    
rna                            | 0.0000   | 0.0000   | 0.0000   | 67      
```
