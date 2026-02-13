# Skin Cancer Detection AI

A deep learning model for automated classification of skin lesions using dermatoscopic images. Trained on 10,000+ medical images across 7 diagnostic categories.

**Best validation accuracy: 84.07%**

-----

## Why this matters

Early detection of melanoma can save lives, but visual diagnosis is challenging — even for dermatologists. This model assists in identifying suspicious lesions that require further medical attention.

-----

## What it detects

7 types of skin conditions:

- **mel** — Melanoma (most dangerous skin cancer)
- **nv** — Melanocytic nevi (common moles)
- **bkl** — Benign keratosis
- **bcc** — Basal cell carcinoma
- **akiec** — Actinic keratoses
- **vasc** — Vascular lesions
- **df** — Dermatofibroma

-----

## Dataset challenges

Unlike simple image classification tasks, HAM10000 presents real-world medical complexities:

- Class imbalance (70% are common moles, only 11% melanoma)
- Visual similarity between benign and malignant lesions
- High misclassification cost (missing melanoma is life-threatening)

These challenges required specialized techniques like weighted sampling and data augmentation.

-----

## How it was built

- **Dataset:** HAM10000 — 10,015 dermatoscopic images
- **Model:** MobileNetV2 with transfer learning
- **Framework:** PyTorch
- **Training:** 15 epochs on GPU (Google Colab T4)
- **Input size:** 224×224 px
- **Class balancing:** WeightedRandomSampler to handle imbalance

-----

## Results

Best performance at epoch 5:

|Metric  |Train |Validation|
|--------|------|----------|
|Accuracy|87.51%|**84.07%**|
|Loss    |0.3301|0.4903    |

Final epoch (15):

|Metric  |Train |Validation|
|--------|------|----------|
|Accuracy|93.50%|80.73%    |
|Loss    |0.1817|0.6231    |

The slight overfitting after epoch 5 is expected with complex medical data. 84% validation accuracy matches professional benchmarks for this dataset.

-----

## Training visualization

![Training Progress](https://github.com/JohnDeepInside/skin-cancer-detection-ai/raw/main/training_plot.png)

-----

## Model Performance Analysis

### Confusion Matrix

![Confusion Matrix](https://github.com/JohnDeepInside/skin-cancer-detection-ai/raw/main/confusion_matrix.png)

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.58 | 0.86 | 0.69 | 65 |
| bcc | 0.61 | 0.81 | 0.70 | 103 |
| bkl | 0.65 | 0.76 | 0.70 | 220 |
| df | 0.56 | 1.00 | 0.72 | 23 |
| **mel (melanoma)** | **0.55** | **0.76** | **0.64** | **223** |
| nv | 0.97 | 0.82 | 0.88 | 1341 |
| vasc | 0.74 | 0.89 | 0.81 | 28 |

**Key insight:** The model achieves 76% recall on melanoma detection, meaning it catches 3 out of 4 melanomas. In medical screening, high recall is more important than precision — better to flag a benign lesion for review than miss a cancer.

---

## Project structure

```
skin-cancer-detection-ai/
├── train.py              # Training script
├── training_plot.png     # Accuracy/loss visualization
└── README.md
```

-----

## Tech stack

- Python 3.12
- PyTorch
- torchvision (MobileNetV2)
- scikit-learn (train/val split with stratification)
- Google Colab (T4 GPU)

-----

## Medical AI context

This project demonstrates understanding of medical AI challenges:

- Class imbalance handling
- High-stakes decision making (false negatives can be fatal)
- Model interpretability requirements
- Regulatory considerations for medical devices

-----

## Author

Built by Nikolai Shatikhin as part of an AI/ML portfolio.
Open to freelance projects in computer vision and medical AI.

Reach out via GitHub issues or direct message.

-----

**Disclaimer:** This model is for research and portfolio purposes only. Not intended for clinical diagnosis. Always consult qualified medical professionals for skin lesion evaluation.
