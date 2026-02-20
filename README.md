# OOD Detection and Neural Collapse

**Deep Learning in Computer Vision — ENSTA Paris / IP Paris, 2025–2026**

---

## Team

| Name | Email | Institution |
|------|-------|-------------|
| Adam Gassem | adam.gassem@ensta.fr | ENSTA Paris / ENS Paris-Saclay |
| Mohamed Amine Arous | mohamed-amine.arous@ensta.fr | ENSTA Paris |

---

## Project Description

This project studies two closely related topics in deep learning:

1. **Out-of-Distribution (OOD) Detection** — given a classifier trained on CIFAR-100, can we detect at inference time whether an input comes from a completely different distribution? We implement and benchmark six OOD scoring methods: MSP, Max Logit, Energy, Mahalanobis, ViM, and NECO.

2. **Neural Collapse** — a geometric phenomenon observed near the end of training, where last-layer features and classifier weights converge to a highly structured configuration (a simplex Equiangular Tight Frame). We track four collapse metrics (NC1–NC4) throughout training and visualize the resulting geometry.

We also investigate how data augmentation affects both OOD performance and the Neural Collapse dynamic.

---

## Code Structure

```
├── train.py                  # Training loop with per-epoch NC tracking
├── evaluate.py               # OOD evaluation pipeline (all 6 methods)
├── plot_nc.py                # NC geometry visualizations
├── plot_visualizations.py    # PCA and feature distribution plots
├── config.py                 # Global configuration (paths, hyperparameters)
│
├── src/
│   ├── model/model.py        # ResNet-18 adapted for 32×32 (no maxpool, 3×3 conv1)
│   │                         # Forward hooks on layer1–layer4, avgpool
│   ├── data/data_loader.py   # CIFAR-100 loaders + 5 OOD datasets
│   │                         # Returns (train_aug, val_clean, test_clean, clean_train)
│   ├── ood/ood_scores.py     # MSP, MaxLogit, Energy, Mahalanobis, ViM, NECO
│   └── nc/nc.py              # NC1–NC5 computation
│
├── data/                     # Datasets (CIFAR-100, CIFAR-10, MNIST, SVHN, DTD, Tiny-ImageNet)
├── checkpoints/              # Saved model weights
└── results/
    ├── with aug/             # Figures and CSVs — augmented model
    └── no aug/               # Figures and CSVs — baseline model
```

---

## How to Run

**Train:**
```bash
python train.py
```

**Evaluate OOD** (all 6 methods, all datasets):
```bash
python evaluate.py
```

**Generate NC geometry plots:**
```bash
python plot_nc.py
```

**Generate PCA / feature visualizations:**
```bash
python plot_visualizations.py
```

Results and figures are saved under `results/with aug/` or `results/no aug/` depending on the `config.py` setting.

---

## Key Results

### OOD Detection (with augmentation, AUROC)

| Method | Near-OOD | Far-OOD |
|--------|----------|---------|
| MSP | 0.796 | 0.763 |
| Max Logit | 0.803 | 0.789 |
| Energy | 0.802 | 0.794 |
| Mahalanobis | 0.612 | 0.671 |
| ViM | 0.730 | 0.777 |
| **NECO** | **0.802** | **0.798** |

### Neural Collapse at Epoch 200

| Metric | Ideal | With aug | No aug |
|--------|-------|----------|--------|
| NC1 ↓ | 0 | 0.308 | 0.015 |
| NC2 ETF dev ↓ | 0 | 0.089 | 0.077 |
| NC3 1−cos ↓ | 0 | 0.027 | 0.009 |
| NC4 agreement ↑ | 1 | 0.9999 | 0.9999 |

---

## Figures

### Training Curves

| With Augmentation | Without Augmentation |
|:-----------------:|:--------------------:|
| ![](results/with%20aug/training_curves.png) | ![](results/no%20aug/training_curves.png) |

---

### NC Metrics Over Training

| With Augmentation | Without Augmentation |
|:-----------------:|:--------------------:|
| ![](results/with%20aug/nc_metrics.png) | ![](results/no%20aug/nc_metrics.png) |

---

### OOD Score Distributions (with augmentation)

| Near-OOD (CIFAR-100 vs Tiny-ImageNet) | Far-OOD (CIFAR-100 vs SVHN) |
|:-------------------------------------:|:----------------------------:|
| ![](results/with%20aug/score_dist_near_ood.png) | ![](results/with%20aug/score_dist_far_ood.png) |

---

### Feature Space (PCA)

| 2D ID Classes | 2D ID vs OOD |
|:-------------:|:------------:|
| ![](results/with%20aug/pca_2d_id_classes.png) | ![](results/with%20aug/pca_2d_id_vs_ood.png) |

---

### NC Geometry

| Class Mean Distances (NC2) | Within-Class Variance (NC1) | NC3 Cosine Alignment |
|:--------------------------:|:---------------------------:|:--------------------:|
| ![](results/with%20aug/class_mean_distances.png) | ![](results/with%20aug/within_class_variance.png) | ![](results/with%20aug/nc3_cosine_alignment.png) |

| NC2 Gram Matrix | NC1 Across Layers | NC5 Cross-Layer CKA |
|:---------------:|:-----------------:|:-------------------:|
| ![](results/with%20aug/nc2_gram_matrix.png) | ![](results/with%20aug/nc1_across_layers.png) | ![](results/with%20aug/nc5_cross_layer.png) |
