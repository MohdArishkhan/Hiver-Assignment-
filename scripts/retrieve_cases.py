import sys
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

KB_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "spotify_knowledge_base.csv"
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("Loading knowledge base...")

df = pd.read_csv(KB_FILE)

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
)

df["spotify_reply"] = (
    df["spotify_reply"]
    .fillna("")
    .astype(str)
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=30000
)

customer_vectors = vectorizer.fit_transform(
    df["customer_message"]
)


# ============================================================
# RETRIEVE SIMILAR HISTORICAL CASES
# ============================================================

def retrieve_cases(
    query,
    top_k=5,
    exclude_tweet_ids=None
):
    """
    Retrieve the most similar historical Spotify
    customer-support cases.

    Parameters
    ----------
    query : str
        Incoming customer message.

    top_k : int
        Number of historical cases to return.

    exclude_tweet_ids : list or None
        Tweet IDs that must not be retrieved.
        Used to prevent evaluation leakage when the
        evaluation example exists in the historical KB.
    """

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()

    # --------------------------------------------------------
    # Prevent exact self-retrieval / evaluation leakage
    # --------------------------------------------------------

    if exclude_tweet_ids:

        for tweet_id in exclude_tweet_ids:

            if pd.isna(tweet_id):
                continue

            matches = df.index[
                df["customer_tweet_id"]
                .astype(str)
                .eq(str(tweet_id))
            ]

            for index in matches:
                similarities[index] = -1.0

    # --------------------------------------------------------
    # Get top matches
    # --------------------------------------------------------

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        # Don't return excluded cases
        if similarities[index] < 0:
            continue

        results.append(
            {
                "customer_message":
                    df.iloc[index]["customer_message"],

                "spotify_reply":
                    df.iloc[index]["spotify_reply"],

                "customer_tweet_id":
                    df.iloc[index]["customer_tweet_id"],

                "spotify_reply_id":
                    df.iloc[index]["spotify_reply_id"],

                "similarity":
                    float(similarities[index])
            }
        )

    return results


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        query = " ".join(
            sys.argv[1:]
        )

    else:

        query = input(
            "Enter customer message: "
        ).strip()

    if not query:

        print(
            "Please enter a customer message."
        )

        sys.exit(1)

    results = retrieve_cases(
        query,
        top_k=5
    )

    print(
        "\n=========================================="
    )

    print(
        "        SIMILAR HISTORICAL CASES"
    )

    print(
        "=========================================="
    )

    print("\nCustomer message:")
    print(query)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            "\n------------------------------------------"
        )

        print(
            f"CASE {i}"
        )

        print(
            "------------------------------------------"
        )

        print(
            "Similarity:",
            round(
                result["similarity"],
                4
            )
        )

        print(
            "\nHistorical Customer:"
        )

        print(
            result["customer_message"]
        )

        print(
            "\nHistorical Spotify Reply:"
        )

        print(
            result["spotify_reply"]
        )

    print(
        "\n=========================================="
    )