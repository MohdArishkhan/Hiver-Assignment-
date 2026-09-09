import pandas as pd

FILE = "twcs.csv"

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

print("Dataset loaded:", len(df))

# --------------------------------------------------
# 1. Create lookup: tweet_id -> tweet information
# --------------------------------------------------

tweet_lookup = df.set_index("tweet_id")

# --------------------------------------------------
# 2. Get all SpotifyCares replies
# --------------------------------------------------

spotify = df[
    df["author_id"].astype(str).str.lower() == "spotifycares"
].copy()

print("SpotifyCares replies:", len(spotify))

# --------------------------------------------------
# 3. Find the customer tweet each Spotify reply
# --------------------------------------------------

spotify["customer_tweet_id"] = spotify["in_response_to_tweet_id"]

# Remove replies where parent tweet is missing
spotify = spotify[
    spotify["customer_tweet_id"].notna()
].copy()

# tweet IDs are sometimes read as floats, so convert carefully
spotify["customer_tweet_id"] = spotify["customer_tweet_id"].astype("int64")

# --------------------------------------------------
# 4. Look up customer tweets
# --------------------------------------------------

customer = tweet_lookup.reindex(
    spotify["customer_tweet_id"]
)

spotify["customer_message"] = customer["text"].values
spotify["customer_author_id"] = customer["author_id"].values
spotify["customer_created_at"] = customer["created_at"].values

# --------------------------------------------------
# 5. Keep useful columns
# --------------------------------------------------

result = spotify[
    [
        "customer_tweet_id",
        "tweet_id",
        "customer_author_id",
        "customer_created_at",
        "created_at",
        "customer_message",
        "text"
    ]
].copy()

result = result.rename(
    columns={
        "tweet_id": "spotify_reply_id",
        "text": "spotify_reply"
    }
)

# --------------------------------------------------
# 6. Remove missing customer messages
# --------------------------------------------------

result = result[
    result["customer_message"].notna()
]

# --------------------------------------------------
# 7. Save
# --------------------------------------------------

result.to_csv(
    "spotify_customer_reply_pairs.csv",
    index=False
)

print("\nDone!")

print("Customer → Spotify reply pairs:", len(result))

print("\nExample conversations:\n")

for _, row in result.head(10).iterrows():

    print("=" * 80)

    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nSPOTIFYCARES:")
    print(row["spotify_reply"])