# Brain Tumor MRI Classification Report

## 1. Project Overview

This project studies **4-class brain tumor MRI classification** using a PyTorch-based transfer learning pipeline built on top of a pretrained **ResNet50** backbone.

The four target classes are:

- glioma
- meningioma
- notumor
- pituitary

The original goal was to build a high-accuracy image classification model for the task. After establishing a strong baseline, the project was extended with additional experiments to analyze how classifier design and loss design affect performance, especially for the **glioma** class.

---

## 2. Motivation

The baseline model achieved strong overall performance, but repeated experiments showed that **glioma recall** was consistently lower than the performance on other classes. In other words, the model often identified glioma correctly when it predicted it, but it still missed a nontrivial number of glioma samples.

This observation motivated further experiments in two directions:

1. changing the classifier head
2. changing the training loss to place slightly more emphasis on glioma

The main goal of the extended experiments was not simply to improve overall accuracy, but to see whether the model could improve **glioma recall** without damaging the overall classification quality.

---

## 3. Dataset

The dataset is organized into two top-level folders:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The `Training/` folder is further split into:

- training set
- validation set

The `Testing/` folder is reserved for final evaluation only.

### Class order

The class order used in the project is:

```python
['glioma', 'meningioma', 'notumor', 'pituitary']
```

---

## 4. Data Processing and Augmentation

### Training preprocessing

Training images are processed with:

- resize to 256 × 256
- random horizontal flip
- random rotation
- random affine transformation
- conversion to tensor
- normalization using ImageNet mean and standard deviation

### Validation and test preprocessing

Validation and test images are processed with:

- resize
- tensor conversion
- ImageNet normalization

### Rationale

The project uses standard transfer learning preprocessing for pretrained ResNet models. Training augmentation is intended to improve robustness to moderate changes in image position and orientation. Validation and test preprocessing are kept deterministic to ensure stable evaluation.

---

## 5. Model and Training Setup

## Backbone

The core model is based on:

- pretrained `torchvision` ResNet50

## Training strategy

A **two-stage training procedure** is used:

### Stage 1
Freeze the backbone and train only the classifier head.

### Stage 2
Unfreeze the last part of the network and fine-tune the model.

This staged setup is used to stabilize transfer learning and avoid damaging pretrained features too early in training.

---

## 6. Methods Compared

Three main methods were tested.

### 6.1 `linear`

This is the baseline model:

- pretrained ResNet50 backbone
- linear classification head
- standard cross-entropy loss

### 6.2 `deep_mlp`

This variant replaces the simple linear head with a deeper multilayer perceptron head.

The motivation was to test whether a more expressive classifier head would improve the classification boundary for difficult classes such as glioma.

### 6.3 `linear_weighted_ce_gli_1.3`

This method keeps the linear head but modifies the loss function by applying a mild class weight to glioma:

```python
[1.3, 1.0, 1.0, 1.0]
```

This means glioma errors are penalized slightly more during training.

The intention is to encourage the model to become less conservative on glioma predictions and improve glioma recall.

---

## 7. Evaluation Metrics

The project evaluates models using:

- test accuracy
- classification report
  - precision
  - recall
  - F1-score
- confusion matrix

Special attention is given to:

- **glioma recall**
- **macro-average performance**
- stability across multiple random seeds

---

## 8. Random Seed Experiments

Experiments were run with:

- seed 1
- seed 2
- seed 3

Different seeds affect:

- train/validation split
- dataloader shuffle order
- data augmentation randomness
- classifier initialization
- dropout randomness

This is why multi-seed evaluation is important. A single run can be informative, but multi-seed results provide a more reliable estimate of method quality and stability.

---

## 9. Baseline Results: `linear`

### Per-seed results

| Method | Seed | Test Accuracy | Glioma Recall |
|---|---:|---:|---:|
| linear | 1 | 0.9238 | 0.77 |
| linear | 2 | 0.9056 | 0.76 |
| linear | 3 | 0.9344 | 0.81 |

### Average result

- **Average test accuracy:** 0.9213
- **Average glioma recall:** 0.7800

### Observations

The linear head is a strong baseline. It achieves good overall classification performance, especially on:

- notumor
- pituitary

However, across seeds, the most consistent weakness is **glioma recall**. Glioma samples are often confused with:

- meningioma
- notumor

This makes glioma the most important failure mode in the baseline.

---

## 10. Deep Classifier Head Results: `deep_mlp`

The deeper MLP head was introduced as an architectural experiment.

### Observation

Under the current training setup, the deep MLP head **underperformed** the plain linear baseline.

### Interpretation

This suggests that, in this project:

- a more complex classifier head does not automatically improve transfer learning performance
- the deeper head is harder to optimize
- higher classifier complexity may not be the right solution for the main error pattern

### Conclusion on `deep_mlp`

The deep MLP variant was useful as an ablation experiment, but it was not selected as the final method.

---

## 11. Weighted Loss Results: `linear_weighted_ce_gli_1.3`

### Per-seed results

| Method | Seed | Test Accuracy | Glioma Recall |
|---|---:|---:|---:|
| linear_weighted_ce_gli_1.3 | 1 | 0.9275 | 0.80 |
| linear_weighted_ce_gli_1.3 | 2 | 0.9313 | 0.81 |
| linear_weighted_ce_gli_1.3 | 3 | 0.9381 | 0.83 |

### Average result

- **Average test accuracy:** 0.9323
- **Average glioma recall:** 0.8133

### Comparison with baseline

Compared with the plain linear baseline:

- average test accuracy improved from **0.9213** to **0.9323**
- average glioma recall improved from **0.7800** to **0.8133**

### Interpretation

This is an important result because the weighted loss improves the class that originally caused the largest difficulty, while also improving average overall accuracy.

The weighted CE method does not simply trade away one class for another in a harmful way. Instead, it appears to shift the training objective in a useful and controlled direction.

---

## 12. Key Experimental Findings

### 12.1 The linear head is a strong baseline

A simple linear classifier on top of a pretrained ResNet50 already provides strong performance.

### 12.2 A deeper classifier head is not necessarily better

The deep MLP head performed worse than the linear baseline under the current setup. This shows that increasing classifier complexity is not always the best way to improve classification quality.

### 12.3 Mild class weighting is effective

A modest weighting of the glioma class improved both:

- average test accuracy
- glioma recall

This makes it the best single-model method tested in the project.

### 12.4 Glioma remains the most challenging class

Even after improvement, glioma remains the hardest class compared with the other three categories. However, the weighted loss reduces this weakness in a meaningful way.

---

## 13. Representative Result Interpretation

The most important pattern observed in the confusion matrices is that glioma samples are often confused with:

- meningioma
- notumor

The weighted CE method reduces this problem by increasing the number of correctly classified glioma samples. In multiple runs, the model became more willing to identify glioma without causing severe performance collapse on the other categories.

This is exactly the type of behavior desired from a targeted loss adjustment.

---

## 14. Final Conclusion

The final conclusion of this project is:

1. A ResNet50-based transfer learning pipeline is effective for 4-class brain tumor MRI classification.
2. A simple linear head provides a strong and stable baseline.
3. Replacing the linear head with a deeper MLP does not improve results under the current setup.
4. A mild class-weighted cross-entropy loss with weights `[1.3, 1.0, 1.0, 1.0]` consistently improves glioma recall and increases average test accuracy across multiple random seeds.
5. The weighted CE variant is the best single-model method tested in this project.

---

## 15. Final Recommended Method

The recommended final method for this repository is:

- **Method:** `linear_weighted_ce_gli_1.3`
- **Backbone:** pretrained ResNet50
- **Head:** linear
- **Loss:** class-weighted cross-entropy
- **Weights:** `[1.3, 1.0, 1.0, 1.0]`

---

## 16. Future Work

Possible future extensions include:

- foreground cropping or black-border removal
- test-time augmentation
- multi-seed model ensemble
- further tuning of class weights
- additional medical image preprocessing experiments

---

## 17. Summary for GitHub / Resume Context

This project demonstrates:

- end-to-end PyTorch image classification workflow development
- transfer learning and staged fine-tuning
- modular ML code organization
- multi-seed evaluation
- ablation-style method comparison
- targeted performance improvement through loss reweighting
- interpretation of class-specific error patterns
