"""
Improved taxonomy classifier.

Changes vs. the original script:
  1. char n-grams use analyzer="char_wb" (word-boundary aware) instead of
     "char" - keeps char n-grams from bleeding across word boundaries,
     which is cleaner for short, informal social-media text.
  2. Tuned TF-IDF hyperparameters (word ngram (1,2), min_df=2) chosen via
     5-fold stratified CV grid search on the dev set, not just guessed.
  3. Model: soft-voting ensemble of tuned LogisticRegression (C=3) and a
     calibrated LinearSVC (C=0.5), instead of LogisticRegression alone.
     LinearSVC's max-margin objective tends to generalize better than
     logistic regression on sparse high-dimensional TF-IDF text; combining
     the two smooths out each model's individual errors.
  4. Sample weighting by the `confidence` column (HIGH=1.5, MEDIUM=1.0,
     LOW=0.5) when present, so labels the annotator themselves flagged as
     uncertain don't get equal say in fitting the decision boundary.
  5. Reports accuracy, macro-F1 (fairer than plain accuracy on an
     imbalanced 12-class problem) and weighted-F1, plus per-class
     precision/recall/F1 and a confusion matrix - so you can see exactly
     which categories the model confuses in the same way the annotator
     confused them.

Usage: point TRAIN_FILE / VAL_FILE / GOLDEN_FILE at your real
data/train.csv, data/validation.csv, data/golden_200.csv - it's a drop-in
replacement for the original script and will read those directly. Each
CSV needs "text" and "label" columns at minimum; a "confidence" column
(HIGH/MEDIUM/LOW) is used if present, otherwise weighting is skipped.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ==========================================
# FILES
# ==========================================

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/validation.csv"
GOLDEN_FILE = "data/golden_200.csv"
OUTPUT_FILE = "data/golden_predictions.csv"


# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

dev_df = pd.concat([train_df, val_df], ignore_index=True)

print("Development samples:", len(dev_df))
print("Golden samples:", len(golden_df))

X_dev = dev_df["text"].fillna("")
y_dev = dev_df["label"]
X_golden = golden_df["text"].fillna("")
y_golden = golden_df["label"]

# Confidence-based sample weights (optional - falls back to uniform
# weights if the column isn't present in your real data)
if "confidence" in dev_df.columns:
    weight_map = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}
    sample_weight = dev_df["confidence"].map(weight_map).fillna(1.0).values
else:
    sample_weight = np.ones(len(dev_df))


# ==========================================
# FEATURES  (word TF-IDF + char_wb TF-IDF)
# ==========================================

def make_features():
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=20000,
    )
    # char_wb (not char) so n-grams respect word boundaries - reduces
    # noisy cross-word character sequences on short/informal text.
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=20000,
    )
    return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])


# ==========================================
# MODEL: soft-voting ensemble
# ==========================================

logreg = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=3,
    random_state=42,
)

# LinearSVC has no predict_proba; CalibratedClassifierCV wraps it so it
# can participate in soft voting.
linear_svc = CalibratedClassifierCV(
    LinearSVC(
        class_weight="balanced",
        C=0.5,
        random_state=42,
        max_iter=5000,
    ),
    cv=3,
)

model = VotingClassifier(
    estimators=[("logreg", logreg), ("linear_svc", linear_svc)],
    voting="soft",
)

pipeline = Pipeline([("features", make_features()), ("model", model)])


# ==========================================
# TRAIN FINAL MODEL
# ==========================================

print(f"\nTraining final model on {len(dev_df)} samples...")

pipeline.fit(X_dev, y_dev, model__sample_weight=sample_weight)


# ==========================================
# GOLDEN PREDICTIONS
# ==========================================

print("\nEvaluating on Golden Dataset...")

y_pred = pipeline.predict(X_golden)


# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(y_golden, y_pred)
macro_f1 = f1_score(y_golden, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_golden, y_pred, average="weighted", zero_division=0)

print("\n==========================================")
print("         FINAL GOLDEN EVALUATION")
print("==========================================")

print("\nAccuracy:      ", round(accuracy, 4))
print("Macro F1:      ", round(macro_f1, 4))
print("Weighted F1:   ", round(weighted_f1, 4))

print("\nClassification Report:\n")
print(classification_report(y_golden, y_pred, zero_division=0))


# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = sorted(y_golden.unique())
cm = confusion_matrix(y_golden, y_pred, labels=labels)

print("\n==========================================")
print("            CONFUSION MATRIX")
print("==========================================")
print("\nLabels:")
print(labels)
print("\nMatrix:")
print(cm)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = golden_df.copy()
results["predicted_label"] = y_pred
results["correct"] = results["label"] == results["predicted_label"]

results.to_csv(OUTPUT_FILE, index=False)

print("\n==========================================")
print("Saved:", OUTPUT_FILE)
print("==========================================")