import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


# ==========================================
# LOAD DATA
# ==========================================

TRAIN_FILE = "data/train.csv"

df = pd.read_csv(TRAIN_FILE)

X = df["text"].fillna("")
y = df["label"]

print("Total training samples:", len(df))


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
# COMBINE FEATURES
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

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("features", features),
    ("model", model)
])


# ==========================================
# CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning 5-fold cross-validation...")

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=[
        "accuracy",
        "f1_macro",
        "f1_weighted"
    ],
    n_jobs=-1
)


# ==========================================
# RESULTS
# ==========================================

print("\n==========================================")
print("       5-FOLD CROSS-VALIDATION")
print("==========================================")

print("\nAccuracy per fold:")
print(scores["test_accuracy"])

print(
    "\nMean Accuracy:",
    round(scores["test_accuracy"].mean(), 4)
)

print(
    "Std Accuracy:",
    round(scores["test_accuracy"].std(), 4)
)

print("\nMacro F1 per fold:")
print(scores["test_f1_macro"])

print(
    "\nMean Macro F1:",
    round(scores["test_f1_macro"].mean(), 4)
)

print(
    "Std Macro F1:",
    round(scores["test_f1_macro"].std(), 4)
)

print("\nWeighted F1 per fold:")
print(scores["test_f1_weighted"])

print(
    "\nMean Weighted F1:",
    round(scores["test_f1_weighted"].mean(), 4)
)

print(
    "Std Weighted F1:",
    round(scores["test_f1_weighted"].std(), 4)
)

print("\n==========================================")
print("DONE")
print("==========================================")