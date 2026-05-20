# AI ER Hospital Receptionist

**Intelligent Triage & Patient Assessment System**  
CSE351: Introduction to Artificial Intelligence — Spring 2026  
College of Engineering

---

## Overview

Emergency rooms worldwide face overcrowding and inconsistent triage decisions. This project builds an AI-powered desktop application that automates triage intake at the ER entrance. Given a patient's demographics, vital signs, chief complaints, and medical history — all collectible in under three minutes — the system simultaneously outputs:

- **Priority classification** — Needs-Immediate-Attention vs. Can-Wait (Model A: Logistic Regression, AUC 0.920)
- **Admission prediction** — likely to be admitted vs. discharged, with a probability score (Model B: Feedforward Neural Network)
- **Department routing** — recommendation from 14 clinical specialties via a rule-based weighted scorer

The system is designed as a **decision support tool**, not a replacement for clinical judgment. Every output includes a confidence score and requires a human to review before any action is taken.

---

## Dataset

We used the **Yale New Haven Hospital (YNHH) Emergency Department dataset** — 560,486 real anonymized ER visits published by Yale University researchers.

- Source: [Kaggle](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data)
- Format: `.rdata` (R binary), loaded via `pyreadr`
- After preprocessing: **369,203 patients**, **232 features**

> The dataset is not included in this repository. Download it from Kaggle before running the training notebook.

---

## Repository Structure

```
CSE351-ER-Receptionist/
│
├── ER_Receptionist.ipynb     # Full training pipeline (data → models)
├── app.py                    # Desktop application (CustomTkinter)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── models/                   # Trained model files (see below)
    ├── model_a_pipeline.pkl  # sklearn Pipeline (StandardScaler + LogisticRegression)
    ├── model_b_nn.keras      # Keras feedforward neural network
    ├── model_b_scaler.pkl    # StandardScaler for Model B
    ├── le_gender.pkl         # LabelEncoder for gender
    └── feature_cols.pkl      # Ordered list of 232 feature column names
```

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
pyreadr
pandas
numpy
scikit-learn
tensorflow
keras
matplotlib
seaborn
joblib
customtkinter
```

> On some systems, CustomTkinter may require `pip install customtkinter --upgrade`.

---

## Running the Application

Make sure the `models/` folder is in the same directory as `app.py`, then:

```bash
python app.py
```

The application window will open. A green dot in the top-right header confirms models loaded successfully. A red dot means the `models/` folder is missing or incomplete.

---

## How to Use the App

1. **Enter patient demographics** — age and gender
2. **Enter vital signs** — heart rate, systolic BP, diastolic BP, respiratory rate, temperature
3. **Describe the main complaint** — free text (e.g. "chest pain", "seizure", "shortness of breath")
4. **Add additional symptoms** — optional free text
5. **Add medical history** — optional (e.g. "diabetes", "hypertension")
6. Click **Analyze Patient ↯**

The right panel displays:
- Priority level and confidence %
- Recommended department
- Likely outcome (Admit / Discharge) and confidence %
- Admission probability bar
- Detected symptom tags
- Bed preparation recommendation

---

## Model Details

| | Model A | Model B |
|---|---|---|
| Task | Priority classification | Disposition prediction |
| Algorithm | Logistic Regression | Feedforward Neural Network |
| Input | 232 features | 232 features |
| Output | Needs-Immediate-Attention / Can-Wait | P(Admit) → Admit / Discharge |
| Test Accuracy | 84.5% | Validated via sanity check |
| AUC | 0.920 | — |
| Baseline improvement | +16.1% over DummyClassifier | +3.45% over Naive Baseline |
| Training data | 295,398 real YNHH patients | 295,398 real + 500,000 synthetic |

**Model B uses two-phase curriculum training:**
- Phase 1: trains on real YNHH records to learn clinical patterns
- Phase 2: fine-tunes on 500K synthetic sparse samples that mirror app inference conditions (resolves distribution shift between dense training records and sparse runtime inputs)

**Admission threshold:** 0.30 (conservative — reducing missed admissions is prioritized over reducing false positives)

---

## Notebook Structure

| Cell | Content |
|---|---|
| 1–2 | Data loading and exploratory analysis |
| 3–5 | Feature selection, missing value imputation, vital sign filtering |
| 6–7 | Label creation and feature matrix construction |
| 8 | Stratified 80/10/10 train/val/test split |
| 9 | Model A — Logistic Regression training |
| 10 | Synthetic data augmentation function |
| 11 | Model B — two-phase neural network training + sanity check |
| 12 | Model serialization (joblib + keras .save()) |
| 13 | Full evaluation: confusion matrices, ROC/AUC, baseline comparison, loss curves, generalization gap |


## Team

| Member|
| Bavly Tony|
| Mario Maged |
| Omar Nasr |
| Youssef Loza |

---
