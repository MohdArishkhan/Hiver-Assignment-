import pandas as pd

FILE = "data/labels_800.csv"

VALID_LABELS = {
    "APP_TECHNICAL_ISSUE",
    "PLAYBACK_ISSUE",
    "PLAYLIST_LIBRARY",
    "MUSIC_CONTENT_AVAILABILITY",
    "ADS_FREE_TIER",
    "PAYMENT_BILLING",
    "FAMILY_STUDENT_PLANS",
    "PREMIUM_SUBSCRIPTION",
    "DOWNLOAD_OFFLINE",
    "ACCOUNT_LOGIN_SECURITY",
    "FEATURE_REQUEST",
    "GENERAL_OTHER",
}

df = pd.read_csv(FILE)

print("========== DATASET VALIDATION ==========\n")

print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\n--- Duplicate IDs ---")
print("Duplicate sample_id:", df["sample_id"].duplicated().sum())

print("\n--- Missing values ---")
print(df[["sample_id", "text", "label"]].isna().sum())

print("\n--- ID range ---")
print("Minimum ID:", df["sample_id"].min())
print("Maximum ID:", df["sample_id"].max())

print("\n--- Number of labels ---")
print("Unique labels:", df["label"].nunique())

print("\n--- Invalid labels ---")
invalid = sorted(set(df["label"].dropna()) - VALID_LABELS)
print(invalid if invalid else "None")

print("\n--- Label distribution ---")
print(df["label"].value_counts())

print("\n--- Confidence distribution ---")
print(df["confidence"].value_counts())

print("\n========== DONE ==========")