# Baseline Evaluation — Span-Based

## 🔍 Configuration
| Parameter | Value |
|---|---|
| **Model** | `Ihor/gliner-biomed-bi-small-v1.0` |
| **Encoder** | `bi` |
| **Dataset** | `jnlpba` |
| **Test file** | `test_dataset_span_bi.json` |
| **Test samples** | `2000` |
| **Batch size** | `8` |
| **Timestamp** | `20260224_184828` |

## 🟢 Evaluation 1 — Label Names (Excluding 'O')
Entity types used: ['cell line', 'cell type', 'dna', 'protein', 'rna']

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.4543 | 0.5061 | **0.4754** |
| **Micro** | 0.5675 | 0.6414 | **0.6022** |

| Timing | Value |
|---|---|
| **Total inference time** | `60.93 s` |
| **Avg iterations/sec** | `4.10` |
| **Samples/sec** | `32.83` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
cell line                      | 0.4545   | 0.4797   | 0.4668   | 271     
cell type                      | 0.6566   | 0.7365   | 0.6943   | 1036    
dna                            | 0.4053   | 0.3370   | 0.3680   | 546     
protein                        | 0.5824   | 0.6939   | 0.6333   | 2581    
rna                            | 0.1727   | 0.2836   | 0.2147   | 67      
```

## 🔵 Evaluation 2 — Label Descriptions (Excluding 'O')
Entity types used: extended descriptions from label2desc.json

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.2523 | 0.0646 | **0.0935** |
| **Micro** | 0.2568 | 0.0335 | **0.0593** |

| Timing | Value |
|---|---|
| **Total inference time** | `69.56 s` |
| **Avg iterations/sec** | `3.59` |
| **Samples/sec** | `28.75` |

```
Label                          | Prec.    | Rec.     | F1       | Supp.   
---------------------------------------------------------------------------
cell line                      | 0.4694   | 0.1697   | 0.2493   | 271     
cell type                      | 0.2917   | 0.0068   | 0.0132   | 1036    
dna                            | 0.2402   | 0.1007   | 0.1419   | 546     
protein                        | 0.2216   | 0.0159   | 0.0296   | 2581    
rna                            | 0.0385   | 0.0299   | 0.0336   | 67      
```
