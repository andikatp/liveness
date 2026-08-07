# Model Performance Metrics

Benchmark results on CelebA Spoof (70k+ test samples).

## Current Best

### Regular Model (FP32)

| Metric | Value |
|:-------|:-----:|
| **Overall Accuracy** | **98.20%** |
| Real Accuracy | 97.58% |
| Spoof Accuracy | 98.73% |
| **ROC-AUC** | **0.9984** |
| **Average Precision** | **0.9987** |

#### Visualizations

<div align="center">
  <img src="../assets/results/metrics/current_best/conf_matrix.png" width="49%" alt="Confusion Matrix" />
  <img src="../assets/results/metrics/current_best/roc_curve.png" width="49%" alt="ROC Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics/current_best/pr_curve.png" width="60%" alt="Precision-Recall Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics/current_best/confidence_dist.png" width="60%" alt="Confidence Distribution" />
</div>

---

### Quantized Model (INT8)

| Metric | Value |
|:-------|:-----:|
| **Overall Accuracy** | **98.20%** |
| Real Accuracy | 97.55% |
| Spoof Accuracy | 98.73% |
| **ROC-AUC** | **0.9984** |
| **Average Precision** | **0.9987** |

#### Visualizations

<div align="center">
  <img src="../assets/results/metrics_quant/current_best/conf_matrix.png" width="49%" alt="Confusion Matrix (Quantized)" />
  <img src="../assets/results/metrics_quant/current_best/roc_curve.png" width="49%" alt="ROC Curve (Quantized)" />
</div>

<div align="center">
  <img src="../assets/results/metrics_quant/current_best/pr_curve.png" width="60%" alt="Precision-Recall Curve (Quantized)" />
</div>

<div align="center">
  <img src="../assets/results/metrics_quant/current_best/confidence_dist.png" width="60%" alt="Confidence Distribution (Quantized)" />
</div>

---

## Previous Best

### Regular Model (FP32)

| Metric | Value |
|:-------|:-----:|
| **Overall Accuracy** | **97.80%** |
| Real Accuracy | 98.16% |
| Spoof Accuracy | 97.50% |
| **ROC-AUC** | **0.9978** |
| **Average Precision** | **0.9981** |

#### Visualizations

<div align="center">
  <img src="../assets/results/metrics/previous_best/conf_matrix.png" width="49%" alt="Previous Best: Confusion Matrix" />
  <img src="../assets/results/metrics/previous_best/roc_curve.png" width="49%" alt="Previous Best: ROC Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics/previous_best/pr_curve.png" width="60%" alt="Previous Best: Precision-Recall Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics/previous_best/confidence_dist.png" width="60%" alt="Previous Best: Confidence Distribution" />
</div>

---

### Quantized Model (INT8)

| Metric | Value |
|:-------|:-----:|
| **Overall Accuracy** | **97.79%** |
| Real Accuracy | 98.15% |
| Spoof Accuracy | 97.49% |
| **ROC-AUC** | **0.9978** |
| **Average Precision** | **0.9981** |

#### Visualizations

<div align="center">
  <img src="../assets/results/metrics_quant/previous_best/conf_matrix.png" width="49%" alt="Previous Best Quantized: Confusion Matrix" />
  <img src="../assets/results/metrics_quant/previous_best/roc_curve.png" width="49%" alt="Previous Best Quantized: ROC Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics_quant/previous_best/pr_curve.png" width="60%" alt="Previous Best Quantized: Precision-Recall Curve" />
</div>

<div align="center">
  <img src="../assets/results/metrics_quant/previous_best/confidence_dist.png" width="60%" alt="Previous Best Quantized: Confidence Distribution" />
</div>

---

## Notes

**Improvements over previous best:**
- Accuracy: 97.80% → 98.20% (+0.40%)
- ROC-AUC: 0.9978 → 0.9984
- AP: 0.9981 → 0.9987

**Quantization:**
- No accuracy drop after INT8 quantization
- File size reduced to 600 KB (67% smaller)
- Same ROC-AUC and AP scores
