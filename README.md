# Mantis Vision

**Compact, gradient-free visual feature learning for handwritten digit recognition**

Mantis Vision is a vision system that uses evolutionary optimization to discover compact convolutional feature detectors for image classification. On MNIST, the system achieves **98.55% test accuracy** with a **40 KB inference-time model**, using no gradient descent or backpropagation.

The approach focuses on learning discriminative visual features directly, followed by a closed-form linear classifier. The resulting model emphasizes compactness, interpretability, and fast convergence rather than architectural depth.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **98.55%** |
| Model Size | **40 KB** |
| Parameters | ~50,000 |
| Neurons | 200 |
| Training Method | Evolutionary optimization |
| Gradients Required | None |

### Per-Digit Accuracy

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 99.5% | 99.2% | 98.5% | 98.5% | 97.3% | 98.7% | 97.6% | 98.2% | 98.4% | 97.4% |

---

## Analysis and Visualizations

- **Feature Embedding (t-SNE)**  
  Learned features form well-separated clusters for all ten digit classes, indicating strong linear separability.

- **Confusion Matrix**  
  Most errors occur between visually similar digits, consistent with human and model-level ambiguity.

- **Per-Neuron Discriminative Power**  
  Individual neurons contribute unevenly to classification. Evolution favors neurons with high class-separation scores rather than uniform participation.

- **Feature Correlation**  
  Low inter-feature correlation suggests that the evolved detectors capture complementary visual structure rather than redundant patterns.

---

## Architecture

```
Input Image (28 × 28)
        ↓
Evolved Convolutional Detectors
(200 neurons, mixed activation functions)
        ↓
Multi-Shape Spatial Pooling
(e.g., 3×3, 4×2, 2×4)
        ↓
Linear Classifier
(closed-form least-squares solution)
        ↓
Output (10 digit classes)
```

---

## Key Design Principles

- **No backpropagation**  
  Feature detectors are evolved rather than trained via gradients.

- **Mixed activation functions**  
  Each neuron independently selects from multiple nonlinearities.

- **Spatial pooling as representation**  
  Multiple pooling geometries are combined to encode spatial structure.

- **Closed-form classification**  
  The final classifier is solved analytically, with no iterative optimization.

---

## Activation Distribution

| Activation | Count | % |
|---|---|---|
| sin | 46 | 23.0% |
| square | 38 | 19.0% |
| cos | 29 | 14.5% |
| relu | 28 | 14.0% |
| tanh | 27 | 13.5% |
| abs | 25 | 12.5% |
| leaky_relu | 4 | 2.0% |
| gaussian | 3 | 1.5% |

---

## Comparison (Inference-Time Models)

| Model | Parameters | Accuracy | Gradients | Size |
|---|---|---|---|---|
| **Mantis Vision** | ~50K | 98.55% | No | **40 KB** |
| LeNet-5 | ~60K | ~99.0% | Yes | ~250 KB |
| MLP (784-300-100) | ~266K | ~98.4% | Yes | ~1 MB |
| SVM (RBF) | — | ~98.6% | No | ~50 MB |
| Random Forest | — | ~97.0% | No | ~10 MB |

---

## Implementation

### Requirements
- torch  
- torchvision  
- numpy  
- scikit-learn  
- matplotlib  

### Running the Demo
```bash
python demo.py
```

---

## Output Structure

```
output/
├── 01_tsne.png
├── 02_confusion_matrix.png
├── 03_per_class_accuracy.png
├── 04_neuron_activations.png
├── 05_neuron_fisher.png
├── 06_reg_sweep.png
├── 07_feature_correlation.png
├── report.json
└── log.txt
```

---

## Method Overview

Mantis Vision uses evolutionary optimization to discover effective visual features:

1. Initialization — Candidate convolutional detectors are randomly generated  
2. Evaluation — Detectors are scored using class-separability criteria  
3. Selection — High-performing detectors are retained  
4. Variation — Selected detectors are mutated and recombined  
5. Iteration — The process repeats until convergence  

---

## Citation

```bibtex
@software{mantis2026,
  title  = {Mantis Vision: Evolved Feature Detectors for Image Classification},
  author = {Payton Miller},
  year   = {2026},
  url    = {https://github.com/...}
}
```
