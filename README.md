# 📧 Email & SMS Spam Detection using Natural Language Processing (NLP)

A comprehensive, production-grade NLP binary classification pipeline designed to distinguish spam messages from legitimate (ham) messages using text normalization, TF-IDF feature extraction, and machine learning models.

---

## 🎯 Project Overview

This project implements an end-to-end Machine Learning and Natural Language Processing workflow using the classic **SMS Spam Collection Dataset** (5,572 labeled records) from the UCI Machine Learning Repository.

### Key Highlights
- **Exploratory Data Analysis (EDA)**: Class distribution analysis (~86.6% ham, ~13.4% spam), message character and word length distributions.
- **Robust Text Preprocessing Pipeline**: Lowercasing, regex cleaning (URLs, emails, HTML tags, punctuation), tokenization, stopword removal (NLTK), and Porter stemming.
- **TF-IDF Feature Extraction**: Sublinear term frequency scaling and unigram + bigram (`ngram_range=(1, 2)`) extraction. Includes comprehensive mathematical explanations.
- **Model Comparison**:
  1. **Multinomial Naive Bayes (`MultinomialNB`)** — Industry standard text classification baseline.
  2. **Logistic Regression (`LogisticRegression`)** — High-precision probabilistic linear classifier.
  3. **Linear Support Vector Classifier (`LinearSVC`)** — High-margin geometric separator.
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC curves, Precision-Recall curves, and annotated Confusion Matrix heatmaps.
- **In-Depth Discussion on Recall**: Detailed examination of the asymmetric costs between False Positives and False Negatives, the trade-off in consumer vs. enterprise security contexts, and threshold tuning.
- **Visualizations**: Side-by-side WordClouds for Spam and Ham words, top-15 discriminative vocabulary frequency charts.
- **Interactive Inference**: Live test function to evaluate custom email/message text with prediction probabilities.

---

## 📁 Repository Structure

```
email_spam_detection/
├── data/
│   └── SMSSpamCollection            # UCI SMS Spam Collection dataset (5,572 labeled messages)
├── notebooks/
│   └── email_spam_detection.ipynb   # Fully executed, commented Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading & class distribution utilities
│   ├── preprocessor.py              # Text cleaning, stopword removal & stemming pipeline
│   ├── model.py                     # TF-IDF vectorization, model training & evaluation
│   └── generate_notebook.py         # Programmatic generator for the complete notebook
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment

Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Standalone Model Pipeline

Run the end-to-end model training, evaluation, and test inferences directly:
```bash
python src/model.py
```

### 3. Launching the Jupyter Notebook

Open and explore the interactive notebook:
```bash
jupyter notebook notebooks/email_spam_detection.ipynb
```

---

## 📊 Performance Benchmark Summary

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.12%** | **98.48%** | **87.25%** | **0.9253** | **0.9849** |
| **Logistic Regression** | **96.77%** | **100.00%** | **75.84%** | **0.8626** | **0.9855** |
| **Linear SVM (LinearSVC)** | **98.39%** | **97.12%** | **90.60%** | **0.9375** | **0.9853** |

### Key Findings:
- **Logistic Regression achieved 100% Precision**, meaning **0 legitimate emails were misclassified as spam** (zero False Positives). This is critical in consumer inbox environments where missing important emails is intolerable.
- **Linear SVM achieved the highest F1-Score (0.9375) and Recall (90.60%)**, effectively capturing the largest share of spam messages while maintaining 97.12% precision.
- **Multinomial Naive Bayes** demonstrated fast training and robust generalization across all metrics (>98% accuracy and precision).

---

## 🔍 Why is Recall Particularly Important for Spam Detection?

In spam classification:
$$\text{Recall}_{\text{spam}} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

- **The Threat Landscape**: Modern spam frequently carries malicious links, phishing forms, and zero-day malware. A single **False Negative** entering an employee's inbox can lead to corporate breaches or ransomware infection.
- **The Trade-Off**: While high recall ensures that no threats slip past, an overly aggressive filter risks **False Positives** (routing vital emails to junk).
- **Solution**: Practical systems use **confidence thresholding** and tiered quarantine (e.g., direct block for $\ge 0.85$, warning banner for $0.40 - 0.85$, clean inbox for $< 0.40$).
