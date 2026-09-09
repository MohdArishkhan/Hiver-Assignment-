import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# FILES
# ==========================================

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/validation.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "logistic_word_char.pkl"
)


# ==========================================
# LOAD DATA
# ==========================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ==========================================
# INPUT / TARGET
# ==========================================

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]


# ==========================================
# WORD + CHARACTER TF-IDF
# ==========================================

print("\nCreating Word + Character TF-IDF...")


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=20000
)


char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=20000
)


features = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


X_train_features = features.fit_transform(X_train)
X_val_features = features.transform(X_val)


print(
    "Combined feature matrix:",
    X_train_features.shape
)


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_features,
    y_train
)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_val_features)


# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

print("\n==========================================")
print("      IMPROVED VALIDATION RESULTS")
print("==========================================")

print(
    "\nAccuracy:",
    round(accuracy, 4)
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    features,
    os.path.join(
        MODEL_DIR,
        "word_char_features.pkl"
    )
)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    "Model:",
    MODEL_FILE
)

print(
    "Features:",
    os.path.join(
        MODEL_DIR,
        "word_char_features.pkl"
    )
)