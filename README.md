# DriveSafe-Sign-Detection
<p align='center'>
<img width="800" alt="DriveSafe Traffic Detection" src="https://github.com/user-attachments/assets/252feffe-5f2e-4be9-8cc6-0b14b2d49e19" />
</p>

---
## Overview

**DriveSafe-Sign-Detection** is a deep learning-based traffic sign classification system built on a custom Convolutional Neural Network (CNN) and MobileNetwork V2 Pretrained Model. The system mimics the perception component of Advanced Driver-Assistance Systems (ADAS), accurately detecting and classifying 43 types of German traffic signs from real-world images.

This project was trained and evaluated on the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) with high performance and well-documented outputs.

---

## Project Structure

```
DriveSafe-Sign-Detection/
│
├── data/
│   ├── Meta/
│   ├── Test/
│   ├── Train/
│   ├── Meta.csv
│   ├── Train.csv
│   └── Test.csv
│
├── notebooks/
│   ├── 00_init.ipynb                       # Initialization of Directories
│   ├── 01_data_exploration.ipynb           # Visualize & inspect raw GTSRB data
│   ├── 02_custom_CNN_training.ipynb        # Build and train custom CNN
│   ├── 03_mobilenet_comparison.ipynb       # Load MobileNet and evaluate
│   └── 04_visualize_results.ipynb          # Confusion matrix & performance plots
│
├── src/                                    # Python scripts for modular implementation
│   ├── __init__.py                         # Initialization method Called in Main
│   ├── __00__paths.py                      # Global variable for paths to use
│   ├── __01__data_setup.py                 # Downloading, Preprocess Datasets
│   ├── __02__CNN_model_training.py         # Train, validate Dataset and Model Performing
│   └── __03__CNN_model_evaluation.py       # Model Evaluation (Confusion Matrix - Precision)
│
├── outputs/                             # Python scripts for modular implementation
│   ├── model/
│   ├── figures/
│   └── docs/
├── main.py                          # Optional: CLI or orchestrator for full training/eval
├── README.md                        # Full description, instructions, results
├── requirements.txt                 # Python dependencies
└── .gitignore
```

---

## CNN Architecture

The architecture consists of 4 convolutional layers followed by pooling and dropout layers to avoid overfitting. Finally, two dense layers handle classification into 43 categories.

### Layered Diagram

```
Input Layer: (30x30x3 RGB Image)
  │
  └── Conv2D (32 filters, 5x5, ReLU)
      │
      └── Conv2D (64 filters, 5x5, ReLU)
          │
          └── MaxPooling (2x2)
              │
              └── Dropout (0.15)
                  │
                  └── Conv2D (128 filters, 3x3, ReLU)
                      │
                      └── Conv2D (256 filters, 3x3, ReLU)
                          │
                          └── MaxPooling (2x2)
                              │    
                              └── Dropout (0.20)
                                  │
                                  └── Flatten
                                      │
                                      └── Dense (512 units, ReLU)
                                          │
                                          └── Dropout (0.25)
                                              │
                                              └── Dense (43 units, Softmax)
```

### Hyperparameters

* **Batch Size:** 128
* **Epochs:** 35
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy

---

## Dataset

The model was trained and evaluated on the **German Traffic Sign Recognition Benchmark (GTSRB)**.

* 39,000+ training images across 43 classes
* 12,000+ test images
* Images resized to **30x30** during preprocessing for uniformity and speed

---

## Training & Validation

Training was conducted using an 80-20 split from the training data to evaluate validation accuracy across epochs. Here’s a snapshot of how the model performed:

> <img width="4800" height="1500" alt="training_performance" src="https://github.com/user-attachments/assets/1e29d1be-2424-4af7-94fc-599c6538e63d" />

> `outputs/figures/training_performance.png`

From this plot, we observe steady convergence with minimal overfitting.

---

## Class Distribution

A visual overview of the number of images per traffic sign class in the training set.

> <img width="4170" height="1770" alt="class_distribution" src="https://github.com/user-attachments/assets/2e70289f-d379-43c3-8cd5-27353f4b1d0b" />
> `outputs/figures/class_distribution.png`

This step helps validate the dataset balance and highlights potential bias sources.

---

## Model Evaluation

After training, the model was tested on unseen test data.

### Test Accuracy

**96.23%** accuracy achieved on the test dataset (12,629 samples).

### Classification Report Summary

Precision and recall were consistently high across most classes. A few rare or visually ambiguous classes showed slightly reduced performance.

| Metric | Macro Avg | Weighted Avg |
|--------|-----------|---------------|
| **Precision** | 0.95 | 0.97 |
| **Recall**    | 0.94 | 0.96 |
| **F1-score**  | 0.94 | 0.96 |

> `outputs/docs/classification_report.txt`

### Confusion Matrix

Visual representation of model performance across all 43 classes:

> <img width="3600" height="3000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/690e3230-39e7-4349-804d-a218fabaad2d" />
> `outputs/figures/confusion_matrix.png`

Diagonal dominance indicates high accuracy, with minimal off-diagonal misclassifications.

---

## Model Comparison: CNN vs. MobileNetV2

This section presents a comparative analysis between the custom CNN classifier and a MobileNetV2-based model.

### Macro-Averaged Evaluation Metrics

| Metric          | CNN    | MobileNetV2 |
| --------------- | ------ | ----------- |
| Accuracy        | 96.2%  | 58.9%       |
| Macro Precision | 0.9498 | 0.5433      |
| Macro Recall    | 0.9440 | 0.4793      |
| Macro F1-Score  | 0.9430 | 0.4816      |

---

### Visual Comparison

**Location:** `outputs/figures/comparison_CNN_MobileNet.png`


<p align="center">
<img width="800" alt="comparison_CNN_MobileNet" src="https://github.com/user-attachments/assets/68b155e9-8114-47bf-a92f-cad613611392" />
</p>

This visual illustrates the performance gap between the two models using macro-averaged metrics.

---

### Why MobileNetV2 Achieved Lower Performance

The MobileNetV2 model was used solely as a **frozen feature extractor**. Its convolutional base (pre-trained on ImageNet) remained unmodified during training. Only a custom classification head was trained on top. This design choice led to:

* Limited adaptation to traffic sign characteristics
* Dependence on generic features not tailored to the GTSRB dataset
* A significant performance drop compared to the CNN trained from scratch

This setup was intentional, serving as a baseline for comparison. A fine-tuned or fully-trainable MobileNetV2 could yield better results, but would require more compute and longer training time.

---

Place this section in the README right after individual model evaluations and before the project structure or conclusion.
This was done intentionally for baseline comparison purposes. In contrast, the custom CNN was trained end-to-end on the GTSRB dataset, allowing it to learn task-specific features more effectively.

## Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/your-username/DriveSafe-Sign-Detection.git
cd DriveSafe-Sign-Detection
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the pipeline**

```bash
python main.py
```

---

## Output Directory Structure

```plaintext
outputs/
├── model/
│   └── custom_traffic_classifier.h5
│
├── figures/
│   ├── class_distribution.png
│   ├── training_performance.png
│   └── confusion_matrix.png
│
└── docs/
    └── classification_report.txt
```

---

## Submission

This project was developed as part of the **Elevvo AI Internship**, showcasing hands-on application of recommendation systems — from data curation to evaluation — with clear insights into model limitations and comparative strengths.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
