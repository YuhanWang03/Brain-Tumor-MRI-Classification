# Brain Tumor MRI Classification with PyTorch

A PyTorch-based medical image classification project for **4-class brain tumor MRI classification** using transfer learning with a pretrained **ResNet50** backbone.

This project includes:

- modular training and evaluation code
- multi-seed experiments
- comparison of different classifier/loss designs
- targeted improvement of **glioma recall** using **class-weighted cross-entropy**

---

## Task

Classify brain MRI images into one of four categories:

- glioma
- meningioma
- notumor
- pituitary

---

## Methods

The project compares the following methods:

- **`linear`**  
  Pretrained ResNet50 + linear classification head

- **`deep_mlp`**  
  Pretrained ResNet50 + deeper MLP classification head

- **`linear_weighted_ce_gli_1.3`**  
  Pretrained ResNet50 + linear head + class-weighted cross-entropy  
  with weights:

```python
[1.3, 1.0, 1.0, 1.0]
```

This weighted setting mildly emphasizes the **glioma** class during training.

---

## Results

The plain linear baseline achieved an average test accuracy of **0.9213** across 3 random seeds.

A mild class-weighted cross-entropy setting (`linear_weighted_ce_gli_1.3`) improved average test accuracy to **0.9323** and increased average glioma recall from **0.7800** to **0.8133**.

For detailed multi-seed comparisons, confusion matrices, and training curves, see `report.md`.

### Summary Table

| Method | Avg Test Accuracy | Avg Glioma Recall |
|---|---:|---:|
| linear | 0.9213 | 0.7800 |
| linear_weighted_ce_gli_1.3 | 0.9323 | 0.8133 |

---

## Project Structure

```text
brain-tumor-mri-classification/
├── data/
├── models/
├── notebooks/
│   ├── demo.ipynb
│   └── compare_models.ipynb
├── outputs/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train_eval.py
│   └── inference.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline

- load MRI images from class-based folders
- apply preprocessing and augmentation
- fine-tune a pretrained ResNet50 model
- evaluate with:
  - test accuracy
  - classification report
  - confusion matrix
- compare methods across multiple seeds

Training uses a **two-stage setup**:

1. train the classifier head
2. fine-tune the backbone

---

## Dataset Format

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

Notes:

- the full dataset is **not included** in this repository
- place the dataset under the `data/` directory before running

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the dataset under:

```text
data/Training
data/Testing
```

### 3. Run the main training notebook

```text
notebooks/demo.ipynb
```

This notebook supports different experiment settings through:

- `HEAD_TYPE`
- `CLASS_WEIGHTS`

It automatically generates a method name and saves outputs accordingly.

### 4. Compare saved models

```text
notebooks/compare_models.ipynb
```

This notebook loads saved checkpoints and compares methods across seeds.

---

## Example Configurations

### Plain linear baseline

```python
HEAD_TYPE = "linear"
CLASS_WEIGHTS = [1, 1, 1, 1]
```

### Weighted CE version

```python
HEAD_TYPE = "linear"
CLASS_WEIGHTS = [1.3, 1, 1, 1]
```

### Deep MLP head

```python
HEAD_TYPE = "deep_mlp"
CLASS_WEIGHTS = [1, 1, 1, 1]
```

---

## Representative Outputs

You can embed saved outputs directly in GitHub Markdown, for example:

```markdown
![Accuracy Curve](outputs/linear_weighted_ce_gli_1.3/seed_1/accuracy_curve.png)
![Loss Curve](outputs/linear_weighted_ce_gli_1.3/seed_1/loss_curve.png)
![Confusion Matrix](outputs/linear_weighted_ce_gli_1.3/seed_1/confusion_matrix_test.png)
```

---

## Why This Project Matters

This project demonstrates:

- end-to-end PyTorch image classification workflow development
- transfer learning and staged fine-tuning
- modular ML code organization
- multi-seed evaluation
- targeted performance improvement through loss reweighting
- model comparison and experimental analysis

---

## Future Extensions

Possible next steps:

- foreground cropping / black-border removal
- test-time augmentation
- multi-seed ensemble
- further tuning of class weights

