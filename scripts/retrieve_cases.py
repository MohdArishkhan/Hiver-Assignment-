import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# FILE
# ==========================================

KB_FILE = "data/spotify_knowledge_base.csv"


# ==========================================
# LOAD KNOWLEDGE BASE
# ==========================================

print("Loading knowledge base...")

df = pd.read_csv(KB_FILE)

df["customer_message"] = df["customer_message"].fillna("")
df["spotify_reply"] = df["spotify_reply"].fillna("")


# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=30000
)

customer_vectors = vectorizer.fit_transform(
    df["customer_message"]
)


# ==========================================
# RETRIEVE SIMILAR CASES
# ==========================================

def retrieve_cases(query, top_k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "customer_message": df.iloc[index]["customer_message"],
            "spotify_reply": df.iloc[index]["spotify_reply"],
            "similarity": float(similarities[index])
        })

    return results


# ==========================================
# MAIN
# ==========================================

if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
else:
    query = input("Enter customer message: ").strip()


results = retrieve_cases(query)


print("\n==========================================")
print("        SIMILAR HISTORICAL CASES")
print("==========================================")

print("\nCustomer message:")
print(query)


for i, result in enumerate(results, start=1):

    print("\n------------------------------------------")
    print(f"CASE {i}")
    print("------------------------------------------")

    print("Similarity:", round(result["similarity"], 4))

    print("\nHistorical Customer:")
    print(result["customer_message"])

    print("\nHistorical Spotify Reply:")
    print(result["spotify_reply"])

print("\n==========================================")