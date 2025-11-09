# Risultati Test Set - Fine-tuned

**Timestamp:** 2025-11-09 17:23:44

**Dataset:** dataset/test_dataset_tokenlevel.json
**Record totali:** 1000
**Record validi:** 1000
**Token valutati:** 32,622

## Metriche aggregate

- **Macro F1:** 0.3902
- **Micro F1:** 0.6293
- **Precision (macro):** 0.3561
- **Recall (macro):** 0.5568
- **Precision (micro):** 0.6293
- **Recall (micro):** 0.6293

## Report per classe

```
              precision    recall  f1-score   support

   cell type     0.2967    0.6884    0.4147      1983
         rna     0.0900    0.2569    0.1333       144
   cell line     0.1609    0.5083    0.2444       539
         dna     0.1708    0.6299    0.2687      1070
     protein     0.4318    0.6278    0.5117      4318
           O     0.9864    0.6296    0.7686     24568

    accuracy                         0.6293     32622
   macro avg     0.3561    0.5568    0.3902     32622
weighted avg     0.8267    0.6293    0.6852     32622

```
## Distribuzione label

| Label | Predette | Reali | Diff |
|-------|----------|-------|------|
| O | 15683 | 24568 | -8885 |
| protein | 6279 | 4318 | +1961 |
| cell type | 4600 | 1983 | +2617 |
| dna | 3946 | 1070 | +2876 |
| cell line | 1703 | 539 | +1164 |
| rna | 411 | 144 | +267 |
