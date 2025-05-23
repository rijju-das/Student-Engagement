# Frame-Level Student Engagement Classification

This repository accompanies the paper:  
**"Enhancing frame-level student engagement classification through knowledge transfer techniques" (Applied Intelligence, 2024)**  
Authors: Riju Das and Soujanya Dev

<<<<<<< HEAD
---

## 📚 Overview

This project presents a pipeline to classify student engagement at the video frame level using facial features. It combines classical machine learning (XGBoost) with deep learning approaches (CNN with transfer learning). The experiments use two datasets: **DAiSEE** (target) and **WACV** (source).

---

## 🗂 Repository Structure

```
Frame-level-student-engagement/
├── Scripts/
│   ├── DataFormatter.py       # Preprocessing and sequence formatting
│   ├── XGB_pred.ipynb         # XGBoost classifier training & evaluation
│   └── Tab_CNN.ipynb          # CNN with transfer learning
└── README.md                  # Documentation and usage
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

=======
> R. Das, and S. Dev, Optimizing Student Engagement Detection using Facial and Behavioral Features, *Neural Computing and Applications*, 2025.

## Overview

This project explores video-based frame-level engagement classification in students using facial features. It combines classical ML (XGBoost) and deep learning (CNN with transfer learning) techniques, evaluated on DAiSEE and WACV datasets.

## Setup Instructions

### 1. Clone Repository
>>>>>>> e531e95246e8914f241edf4454b687718b16316c
```bash
git clone https://github.com/rijju-das/Student-Engagement.git
cd Student-Engagement/Scripts
```

<<<<<<< HEAD
### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost keras tensorflow matplotlib
```

### 3. Prepare Datasets

Create the following directory structure and add your `features.csv` and `labels.csv`:

```
Data/
├── DAiSEE/
│   ├── features.csv
│   └── labels.csv
└── WACV/
    ├── features.csv
    └── labels.csv
```

Each CSV should contain OpenFace-extracted features and frame-level engagement labels.

---

## 🧪 Reproducing Results

### Step 1: Data Preprocessing

Use `DataFormatter.py` to format and split the data:

```python
from DataFormatter import DataFormatter

# For DAiSEE dataset
formatter = DataFormatter(input_dir='../Data/DAiSEE', output_dir='./processed/DAiSEE', seq_len=16)
formatter.create_datasets()

# For WACV dataset
formatter = DataFormatter(input_dir='../Data/WACV', output_dir='./processed/WACV', seq_len=16)
formatter.create_datasets()
```

### Step 2: XGBoost Baseline

Train and evaluate the XGBoost model:

```bash
jupyter nbconvert --to notebook --execute XGB_pred.ipynb
```

### Step 3: CNN with Transfer Learning

Train CNN on WACV and fine-tune on DAiSEE:

```bash
jupyter nbconvert --to notebook --execute Tab_CNN.ipynb
```

---

## 📊 Results

| Model            | Dataset | Accuracy (%) | Notes                            |
|------------------|---------|--------------|----------------------------------|
| XGBoost          | DAiSEE  | xx.xx        | Baseline using tabular features |
| CNN + Transfer   | DAiSEE  | yy.yy        | Pretrained on WACV, fine-tuned on DAiSEE |

> Replace `xx.xx` and `yy.yy` with your actual experimental results.

---

## 📌 Notes on Scripts

- `DataFormatter.py`: Splits data into train/val/test and reshapes for temporal modeling.
- `XGB_pred.ipynb`: Implements an XGBoost model using tabular features.
- `Tab_CNN.ipynb`: Defines and trains a CNN architecture; supports knowledge transfer.

---

## 📖 Citation

If you use this repository in your research, please cite:

```bibtex
@article{das2024enhancing,
  title={Enhancing frame-level student engagement classification through knowledge transfer techniques},
  author={Das, R and Dev, S},
  journal={Applied Intelligence},
  year={2024},
}
```

---

## 👤 Contact

**Riju Das**  
Ph.D. Scholar, University College Dublin  
GitHub: [rijju-das](https://github.com/rijju-das)  
Email: riju.das@ucd.ie

---
=======
### 1. Clone Repository
```bash
pip install numpy pandas scikit-learn xgboost keras tensorflow matplotlib
```
>>>>>>> e531e95246e8914f241edf4454b687718b16316c
