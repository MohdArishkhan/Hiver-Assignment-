import pandas as pd

FILE = "twcs.csv"

# Only load the columns we need
columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

print("Loading dataset...")

df = pd.read_csv(
    FILE,
    usecols=columns,
    low_memory=False
)

print("Total rows:", len(df))

# SpotifyCares is the brand account
spotify = df[
    (df["author_id"].astype(str).str.lower() == "spotifycares")
    |
    (
        df["in_response_to_tweet_id"].notna()
        & df["author_id"].astype(str).str.lower().eq("spotifycares")
    )
]

print("\nSpotifyCares rows:", len(spotify))

print("\nInbound / outbound:")
print(spotify["inbound"].value_counts())

print("\nSample:")
print(
    spotify[
        ["tweet_id", "author_id", "inbound", "text"]
    ].head(20).to_string(index=False)
)