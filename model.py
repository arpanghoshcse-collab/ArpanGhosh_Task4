"""
Model training, evaluation, and inference pipeline for Email/SMS Spam Detection.
Implements TF-IDF vectorization, Multinomial Naive Bayes, Logistic Regression,
and Linear Support Vector Machine (LinearSVC) classifiers.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset
from src.preprocessor import preprocess_text


def prepare_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Loads dataset, applies preprocessing pipeline, and splits into train/test sets.
    """
    df = load_dataset()
    print("Preprocessing messages...")
    df["cleaned_message"] = df["message"].apply(preprocess_text)

    X = df["cleaned_message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, df


def build_vectorizer(
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Constructs a TF-IDF vectorizer configured for text classification.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )


def train_and_evaluate_models(
    X_train_vec, X_test_vec, y_train, y_test
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains MultinomialNB, Logistic Regression, and LinearSVC.
    Evaluates accuracy, precision, recall, F1-score, and ROC-AUC on the spam class.
    """
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Linear SVM (LinearSVC)": LinearSVC(C=1.0, random_state=42),
    }

    results = {}
    metrics_summary = []

    for name, model in models.items():
        # Train
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # ROC-AUC calculation (using probabilities or decision function)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_vec)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test_vec)
        else:
            y_scores = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "report": classification_report(y_test, y_pred, target_names=["Ham", "Spam"]),
        }

        metrics_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Spam)": prec,
            "Recall (Spam)": rec,
            "F1-Score (Spam)": f1,
            "ROC-AUC": auc,
        })

    summary_df = pd.DataFrame(metrics_summary).set_index("Model")
    return results, summary_df


def predict_message(
    message: str,
    model: Any,
    vectorizer: TfidfVectorizer
) -> Dict[str, Any]:
    """
    Classifies a raw single message as spam or ham.
    """
    clean = preprocess_text(message)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    label = "Spam" if pred == 1 else "Ham"

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = probs[pred]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        # Sigmoid approximation of confidence
        confidence = 1 / (1 + np.exp(-score)) if pred == 1 else 1 - (1 / (1 + np.exp(-score)))

    return {
        "original_message": message,
        "cleaned_message": clean,
        "prediction": label,
        "is_spam": bool(pred == 1),
        "confidence": float(confidence) if confidence is not None else None,
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = prepare_data()
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    results, summary_df = train_and_evaluate_models(X_train_vec, X_test_vec, y_train, y_test)
    print("\n--- Model Evaluation Summary ---")
    print(summary_df.round(4))

    # Test sample predictions
    test_messages = [
        "Hey, can you send me the lecture notes from today? Thanks!",
        "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card! Click here to claim your reward immediately!"
    ]
    best_model = results["Multinomial Naive Bayes"]["model"]
    print("\n--- Sample Inferences ---")
    for msg in test_messages:
        res = predict_message(msg, best_model, vectorizer)
        print(f"\nMessage: '{res['original_message']}'")
        print(f"Prediction: {res['prediction']} (Confidence: {res['confidence']:.2%})")
