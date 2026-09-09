import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

def retrieve_cases(text, top_k=TOP_K):

    query_vector = retrieval_vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        kb_vectors
    ).flatten()

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

    # -----------------------------------------
    # High-risk account security
    # -----------------------------------------

    security_keywords = [
        "hacked",
        "hack",
        "stolen",
        "account taken",
        "someone accessed",
        "unauthorized",
        "phishing",
        "scam",
        "password changed"
    ]

    if intent == "ACCOUNT_LOGIN_SECURITY":
        for keyword in security_keywords:
            if keyword in text_lower:
                return (
                    "ESCALATE",
                    "Potential account security or unauthorized-access issue."
                )

    # -----------------------------------------
    # Billing / refund issues
    # -----------------------------------------

    billing_keywords = [
        "refund",
        "charged twice",
        "charged twice",
        "wrong charge",
        "unexpected charge",
        "dispute",
        "money taken"
    ]

    if intent == "PAYMENT_BILLING":
        for keyword in billing_keywords:
            if keyword in text_lower:
                return (
                    "ESCALATE",
                    "Billing or refund issue may require account-specific action."
                )

    # -----------------------------------------
    # Weak retrieval
    # -----------------------------------------

    best_similarity = 0.0

    if retrieved_cases:
        best_similarity = retrieved_cases[0]["similarity"]

    if best_similarity < 0.20:
        return (
            "ESCALATE",
            "Insufficient similarity to historical Spotify support cases."
        )

    # -----------------------------------------
    # Vague messages
    # -----------------------------------------

    vague_phrases = [
        "still the same",
        "no luck",
        "help",
        "it doesn't work",
        "not working"
    ]

    if len(text.split()) <= 5:
        for phrase in vague_phrases:
            if phrase in text_lower:
                return (
                    "ESCALATE",
                    "Message is too vague to safely generate a reliable response."
                )

    # -----------------------------------------
    # Otherwise auto-handle
    # -----------------------------------------

    return (
        "AUTO_HANDLE",
        "Intent is identifiable and similar historical support cases were found."
    )


# =========================================================
# GROUNDED REPLY
# =========================================================

def generate_reply(intent, retrieved_cases):

    if not retrieved_cases:
        return (
            "I’m sorry you’re experiencing this. "
            "Could you share more details so we can look into it?"
        )

    best_reply = retrieved_cases[0]["spotify_reply"]

    # Remove empty replies
    if not best_reply.strip():
        return (
            "Thanks for reaching out. "
            "Could you share a few more details about the issue?"
        )

    return (
        "Based on similar Spotify support cases, "
        "the recommended next step is:\n\n"
        + best_reply
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
    # Reply
    # -----------------------------------------

    if decision == "AUTO_HANDLE":

        reply = generate_reply(
            intent,
            retrieved_cases
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