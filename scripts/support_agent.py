import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from generate_reply import generate_reply

# =========================================================
# FILES
# =========================================================

MODEL_FILE = "models/final_model.pkl"
KB_FILE = "data/spotify_knowledge_base.csv"

TOP_K = 5


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading intent classifier...")

intent_model = joblib.load(MODEL_FILE)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

print("Loading Spotify knowledge base...")

kb = pd.read_csv(KB_FILE)

kb["customer_message"] = (
    kb["customer_message"]
    .fillna("")
    .astype(str)
)

kb["spotify_reply"] = (
    kb["spotify_reply"]
    .fillna("")
    .astype(str)
)


# =========================================================
# RETRIEVAL MODEL
# =========================================================

retrieval_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=30000
)

kb_vectors = retrieval_vectorizer.fit_transform(
    kb["customer_message"]
)


# =========================================================
# INTENT HELPERS
# =========================================================

HIGH_RISK_INTENTS = {
    "ACCOUNT_LOGIN_SECURITY",
    "PAYMENT_BILLING",
}


def predict_intent(text):
    return intent_model.predict([text])[0]


# =========================================================
# RETRIEVE HISTORICAL CASES
# =========================================================

def retrieve_cases(text, top_k=TOP_K, exclude_tweet_ids=None):

    query_vector = retrieval_vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        kb_vectors
    ).flatten()

    # Prevent evaluation leakage.
    # Do not allow the exact customer case being evaluated
    # to be retrieved as its own historical evidence.
    if exclude_tweet_ids:

        for tweet_id in exclude_tweet_ids:

            matching_rows = kb.index[
                kb["customer_tweet_id"].astype(str)
                == str(tweet_id)
            ]

            similarities[matching_rows] = -1.0

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        similarity = float(similarities[index])

        results.append({
            "customer_message": kb.iloc[index]["customer_message"],
            "spotify_reply": kb.iloc[index]["spotify_reply"],
            "similarity": similarity
        })

    return results


# =========================================================
# ESCALATION DECISION
# =========================================================

def decide_escalation(text, intent, retrieved_cases):

    text_lower = text.lower()

    # =====================================================
    # NORMALIZE COMMON HTML ENCODING
    # =====================================================

    text_lower = (
        text_lower
        .replace("&amp;", " and ")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )

    # =====================================================
    # 1. SECURITY RISK
    # =====================================================

    security_signals = [
        "hacked",
        "hack",
        "hacking",
        "stolen account",
        "account taken",
        "someone accessed",
        "someone is using my account",
        "unauthorized",
        "unauthorised",
        "phishing",
        "scam",
        "fraud",
        "password changed",
        "password was changed",
        "locked me out",
        "locked out of my account",
        "can't log in",
        "cannot log in",
        "couldn't log in",
        "could not log in",
        "unable to log in",
        "unable to reset my password",
        "can't reset my password",
        "cannot reset my password",
    ]

    if any(signal in text_lower for signal in security_signals):

        return (
            "ESCALATE",
            "Potential account security or unauthorized-access issue."
        )

    # =====================================================
    # 2. BILLING / PAYMENT RISK
    # =====================================================

    billing_signals = [
        "charged",
        "charged twice",
        "duplicate charge",
        "wrong charge",
        "unexpected charge",
        "fraudulent charge",
        "refund",
        "billing issue",
        "payment failed",
        "payment won't go through",
        "payment won't",
        "money taken",
        "money was taken",
        "bank statement",
        "charged my card",
        "charging my account",
    ]

    if any(signal in text_lower for signal in billing_signals):

        return (
            "ESCALATE",
            "Billing or payment issue may require account-specific action."
        )

    # =====================================================
    # 3. ACCOUNT-SPECIFIC PREMIUM STATUS
    # =====================================================

    premium_state_signals = [
        "premium does not work",
        "premium doesn't work",
        "premium not working",
        "still says free",
        "shows free",
        "changed to free",
        "account has changed to free",
        "premium has stopped working",
        "paid for premium",
        "purchased premium",
        "premium status",
        "subscription status",
    ]

    if any(
        signal in text_lower
        for signal in premium_state_signals
    ):

        return (
            "ESCALATE",
            "Subscription status may require checking the user's account."
        )

    # =====================================================
    # 4. ACCOUNT-SPECIFIC PLAYLIST / LIBRARY DATA
    # =====================================================

    library_signals = [
        "playlist disappeared",
        "playlists disappeared",
        "playlist is gone",
        "playlists are gone",
        "deleted my playlist",
        "deleted my library",
        "lost my playlist",
        "lost my playlists",
        "lost my library",
        "my library disappeared",
        "all my saved music",
        "saved music disappeared",
        "playlists don't sync",
        "playlists do not sync",
        "not syncing between",
        "restore my playlist",
        "restore previous week's playlist",
        "randomly deleting my library",
        "randomly deleted my library",
    ]

    if any(
        signal in text_lower
        for signal in library_signals
    ):

        return (
            "ESCALATE",
            "Account-specific playlist or library data may require support review."
        )

    # =====================================================
    # 5. VAGUE / CONTEXT-FREE MESSAGE
    # =====================================================

    vague_signals = [
        "still the same",
        "no luck",
        "not sure",
        "same issue",
        "no solution yet",
        "please help",
    ]

    if len(text_lower.split()) <= 8:

        if any(
            signal in text_lower
            for signal in vague_signals
        ):

            return (
                "ESCALATE",
                "Message lacks enough actionable context for safe automated handling."
            )

    # =====================================================
    # 6. WEAK HISTORICAL EVIDENCE
    # =====================================================

    best_similarity = 0.0

    if retrieved_cases:
        best_similarity = retrieved_cases[0]["similarity"]

    if best_similarity < 0.20:

        return (
            "ESCALATE",
            "Insufficient similarity to historical Spotify support cases."
        )

    # =====================================================
    # 7. OTHERWISE AUTO-HANDLE
    # =====================================================

    return (
        "AUTO_HANDLE",
        "Intent is identifiable and similar historical support cases were found."
    )
# =========================================================
# MAIN AGENT
# =========================================================

def run_agent(text):

    print("\n==========================================")
    print("          SPOTIFY SUPPORT AGENT")
    print("==========================================")

    print("\nCustomer:")
    print(text)

    # -----------------------------------------
    # Intent
    # -----------------------------------------

    intent = predict_intent(text)

    print("\nPredicted Intent:")
    print(intent)

    # -----------------------------------------
    # Retrieval
    # -----------------------------------------

    retrieved_cases = retrieve_cases(text)

    print("\nTop Historical Match:")
    print(
        round(
            retrieved_cases[0]["similarity"],
            4
        )
    )

    # -----------------------------------------
    # Decision
    # -----------------------------------------

    decision, reason = decide_escalation(
        text,
        intent,
        retrieved_cases
    )

    print("\nDecision:")
    print(decision)

    print("\nReason:")
    print(reason)

    # -----------------------------------------
    #   
    # -----------------------------------------

    if decision == "AUTO_HANDLE":

        reply = generate_reply(
            customer_message=text,
            intent=intent,
            evidence=retrieved_cases[:3]
        )

        print("\nDraft Reply:")
        print(reply)

    else:

        print("\nDraft Reply:")
        print("Human support review recommended.")

    # -----------------------------------------
    # Evidence
    # -----------------------------------------

    print("\nHistorical Evidence:")

    for i, case in enumerate(
        retrieved_cases[:3],
        start=1
    ):

        print(
            f"\nCase {i} "
            f"(similarity={case['similarity']:.4f})"
        )

        print("Customer:")
        print(case["customer_message"])

        print("Spotify:")
        print(case["spotify_reply"])

    print("\n==========================================")

    return {
        "intent": intent,
        "decision": decision,
        "reason": reason,
        "retrieved_cases": retrieved_cases
    }


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:]).strip()
    else:
        message = input(
            "Enter Spotify customer message: "
        ).strip()

    if not message:
        print("Please enter a customer message.")
        sys.exit(1)

    run_agent(message)