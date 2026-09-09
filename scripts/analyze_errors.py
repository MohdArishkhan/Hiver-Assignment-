import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# ==========================================
# FILES
# ==========================================

VAL_FILE = "data/validation.csv"

MODEL_FILE = "models/logistic_regression.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"


# ==========================================
# LOAD
# ==========================================

df = pd.read_csv(VAL_FILE)

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)


# ==========================================
# PREDICT
# ==========================================

X = df["text"].fillna("")
y_true = df["label"]

X_tfidf = vectorizer.transform(X)

y_pred = model.predict(X_tfidf)


# ==========================================
# CREATE ERROR DATAFRAME
# ==========================================

results = df.copy()

results["predicted_label"] = y_pred
results["correct"] = results["label"] == results["predicted_label"]


# ==========================================
# ERROR SUMMARY
# ==========================================

errors = results[results["correct"] == False].copy()

print("\n==========================================")
print("           ERROR ANALYSIS")
print("==========================================")

print("\nTotal validation samples:", len(results))
print("Correct predictions:", results["correct"].sum())
print("Incorrect predictions:", len(errors))

print("\nError rate:", round(len(errors) / len(results), 4))


# ==========================================
# MOST COMMON CONFUSIONS
# ==========================================

print("\n==========================================")
print("       MOST COMMON CONFUSIONS")
print("==========================================")

confusions = (
    errors
    .groupby(["label", "predicted_label"])
    .size()
    .sort_values(ascending=False)
)

for (true_label, predicted_label), count in confusions.items():
    print(f"{true_label} -> {predicted_label}: {count}")


# ==========================================
# PRINT INDIVIDUAL ERRORS
# ==========================================

print("\n==========================================")
print("        MISCLASSIFIED EXAMPLES")
print("==========================================")

for _, row in errors.iterrows():

    print("\n------------------------------------------")
    print("Sample ID:", row["sample_id"])
    print("Text:", row["text"])
    print("True:", row["label"])
    print("Predicted:", row["predicted_label"])
    print("Confidence:", row["confidence"])
    print("Notes:", row["notes"])


# ==========================================
# SAVE ERRORS
# ==========================================

OUTPUT_FILE = "data/validation_errors.csv"

errors.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("Saved error file:")
print(OUTPUT_FILE)
print("==========================================")