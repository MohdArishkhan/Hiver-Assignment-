import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# FILES
# ==========================================

INPUT_FILE = "data/golden_predictions.csv"
OUTPUT_DIR = "reports"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

y_true = df["label"]
y_pred = df["predicted_label"]


# ==========================================
# LABELS
# ==========================================

labels = sorted(df["label"].unique())


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)


# ==========================================
# PLOT
# ==========================================

fig, ax = plt.subplots(
    figsize=(14, 12)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
)

ax.set_title(
    "Spotify Customer Support Intent - Confusion Matrix"
)

plt.tight_layout()


# ==========================================
# SAVE
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print(OUTPUT_FILE)