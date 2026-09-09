import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/labels_800.csv"

GOLDEN_FILE = "data/golden_200.csv"
TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/validation.csv"

RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

print("Original dataset:", len(df))

# --------------------------------------------------
# STEP 1: Prefer HIGH-confidence examples for Golden
# --------------------------------------------------

high_conf = df[df["confidence"] == "HIGH"].copy()
other_conf = df[df["confidence"] != "HIGH"].copy()

# We need 200 golden samples.
# Stratify using labels.
#
# First try selecting from HIGH-confidence samples.
golden, remaining_high = train_test_split(
    high_conf,
    test_size=len(high_conf) - 200,
    stratify=high_conf["label"],
    random_state=RANDOM_STATE
)

# Everything not selected for Golden
remaining = pd.concat(
    [remaining_high, other_conf],
    ignore_index=True
)

# --------------------------------------------------
# STEP 2: Split remaining 600 into train/validation
# --------------------------------------------------

train, validation = train_test_split(
    remaining,
    test_size=120,
    stratify=remaining["label"],
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# STEP 3: Save
# --------------------------------------------------

golden = golden.sort_values("sample_id")
train = train.sort_values("sample_id")
validation = validation.sort_values("sample_id")

golden.to_csv(GOLDEN_FILE, index=False)
train.to_csv(TRAIN_FILE, index=False)
validation.to_csv(VAL_FILE, index=False)

# --------------------------------------------------
# STEP 4: Print statistics
# --------------------------------------------------

print("\n========== SPLIT RESULT ==========")

print("\nGolden:", len(golden))
print("Train:", len(train))
print("Validation:", len(validation))

print("\n--- Golden label distribution ---")
print(golden["label"].value_counts().sort_index())

print("\n--- Train label distribution ---")
print(train["label"].value_counts().sort_index())

print("\n--- Validation label distribution ---")
print(validation["label"].value_counts().sort_index())

print("\n--- Golden confidence ---")
print(golden["confidence"].value_counts())

print("\nFiles created:")
print(GOLDEN_FILE)
print(TRAIN_FILE)
print(VAL_FILE)

print("\n========== DONE ==========")