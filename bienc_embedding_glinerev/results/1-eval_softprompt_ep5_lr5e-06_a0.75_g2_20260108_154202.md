# Evaluation Report
**Date**: 20260108_154202
**Model Type**: Bi-Encoder Soft Prompt

## Configuration
- **Epochs**: 5
- **Learning Rate**: 5e-06
- **Focal Alpha**: 0.75
- **Focal Gamma**: 2
- **Weight Decay**: 0.01
- **Batch Size**: 8

## Metrics
| Metric | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Macro** | 0.4793 | 0.5671 | **0.4991** |
| **Micro** | 0.4814 | 0.6893 | **0.5669** |

## Detailed Metrics by Label
The detailed per-label metrics are available in the console logs.

https://www.kaggle.com/code/alessandrobecci/finetuneglinerbiembedd-injectionintraingliner
### Performance Summary
| Average Type | Precision | Recall | F1-Score |
|:-------------|----------:|-------:|---------:|
| **Macro**    | 0.6542 | 0.7623 | **0.7018** |
| **Micro**    | 0.6873 | 0.8098 | **0.7436** |