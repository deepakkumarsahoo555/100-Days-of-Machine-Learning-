"""
Trains fake-news classifiers on top of TF-IDF features.

Usage:
    python src/train_model.py --data data/news_dataset.csv

Or, to use the Kaggle "Fake and Real News" dataset instead:
    python src/train_model.py --kaggle-fake data/Fake.csv --kaggle-true data/True.csv

Outputs (in models/):
    - vectorizer.joblib     TF-IDF vectorizer fitted on the training data
    - model.joblib          Best-performing classifier
    - metrics.json          Accuracy / precision / recall / F1 for the dashboard
    - dataset_stats.json    Total / real / fake article counts for the dashboard
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_series

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_dataset(path):
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must have 'text' and 'label' columns")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    return df[["text", "label"]]


def load_kaggle_dataset(fake_path, true_path):
    """Loads Kaggle's Fake.csv / True.csv format (columns: title, text, subject, date)."""
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"] = "FAKE"
    true["label"] = "REAL"
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = df.get("title", "").fillna("") + ". " + df.get("text", "").fillna("")
    return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)


def train(df, test_size=0.2):
    print(f"Loaded {len(df)} articles: {df['label'].value_counts().to_dict()}")

    print("Cleaning text...")
    df["clean_text"] = clean_series(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=test_size, random_state=42, stratify=df["label"]
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    best_name, best_model, best_acc, best_metrics = None, None, -1, None

    for name, clf in candidates.items():
        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="FAKE", zero_division=0)
        rec = recall_score(y_test, preds, pos_label="FAKE", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="FAKE", zero_division=0)

        print(f"  {name:<24s} acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

        if acc > best_acc:
            best_name, best_model, best_acc = name, clf, acc
            best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, **{k: round(v * 100, 2) for k, v in best_metrics.items()}}, f, indent=2)

    counts = df["label"].value_counts().to_dict()
    stats = {
        "total": int(len(df)),
        "real": int(counts.get("REAL", 0)),
        "fake": int(counts.get("FAKE", 0)),
    }
    with open(os.path.join(MODELS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved model, vectorizer, metrics.json and dataset_stats.json to {MODELS_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--kaggle-fake", default=None, help="Path to Kaggle Fake.csv")
    parser.add_argument("--kaggle-true", default=None, help="Path to Kaggle True.csv")
    args = parser.parse_args()

    if args.kaggle_fake and args.kaggle_true:
        df = load_kaggle_dataset(args.kaggle_fake, args.kaggle_true)
    elif args.data:
        df = load_dataset(args.data)
    else:
        default_path = os.path.join("data", "news_dataset.csv")
        if not os.path.exists(default_path):
            print("No dataset found. Generating a synthetic sample dataset first...")
            from data.generate_sample_data import generate_dataset
            generate_dataset(n_per_class=400).to_csv(default_path, index=False)
        df = load_dataset(default_path)

    train(df)


if __name__ == "__main__":
    main()
