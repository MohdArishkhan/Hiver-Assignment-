import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ==========================================
# FILES
# ==========================================

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/validation.csv"
GOLDEN_FILE = "data/golden_200.csv"


# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

# Combine training + validation
dev_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

print("Development samples:", len(dev_df))
print("Golden samples:", len(golden_df))


# ==========================================
# DEVELOPMENT DATA
# ==========================================

X_dev = dev_df["text"].fillna("")
y_dev = dev_df["label"]

X_golden = golden_df["text"].fillna("")
y_golden = golden_df["label"]


# ==========================================
# WORD TF-IDF
# ==========================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=20000
)


# ==========================================
# CHARACTER TF-IDF
# ==========================================

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=20000
)


# ==========================================
# FEATURE UNION
# ==========================================

features = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


# ==========================================
# MODEL
# ==========================================

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)


# ==========================================
# PIPELINE
# ==========================================

pipeline = Pipeline([
    ("features", features),
    ("model", model)
])


# ==========================================
# TRAIN FINAL MODEL
# ==========================================

print("\nTraining final model on 600 samples...")

pipeline.fit(
    X_dev,
    y_dev
)


# ==========================================
# GOLDEN PREDICTIONS
# ==========================================

print("\nEvaluating on Golden Dataset...")

y_pred = pipeline.predict(X_golden)


# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(
    y_golden,
    y_pred
)

print("\n==========================================")
print("         FINAL GOLDEN EVALUATION")
print("==========================================")

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:\n")

print(
    classification_report(
        y_golden,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = sorted(y_golden.unique())

cm = confusion_matrix(
    y_golden,
    y_pred,
    labels=labels
)

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
results["correct"] = (
    results["label"] ==
    results["predicted_label"]
)

OUTPUT_FILE = "data/golden_predictions.csv"

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("Saved:")
print(OUTPUT_FILE)
print("==========================================")

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/final_model.pkl")

print("Saved final model: models/final_model.pkl")