from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = ROOT / "data" / "golden_200.csv"
OUTPUT_FILE = ROOT / "data" / "agent_eval.csv"


# =========================================================
# POLICY
# =========================================================

def expected_decision(row):
    text = str(row["text"]).lower()
    intent = row["label"]

    # -----------------------------------------------------
    # Account security
    # -----------------------------------------------------

    security_keywords = [
        "hack",
        "hacked",
        "hacking",
        "fraud",
        "unauthorized",
        "someone",
        "stolen",
        "locked out",
        "can't log in",
        "cannot log in",
        "unable to log in",
        "password",
        "logged out",
        "login",
        "sign in",
    ]

    if intent == "ACCOUNT_LOGIN_SECURITY":
        if any(keyword in text for keyword in security_keywords):
            return "ESCALATE"

    # -----------------------------------------------------
    # Billing
    # -----------------------------------------------------

    billing_keywords = [
        "charged",
        "charging",
        "refund",
        "fraudulent charge",
        "payment",
        "money",
        "billing",
        "card",
        "bank statement",
        "premium does not work",
    ]

    if intent == "PAYMENT_BILLING":
        if any(keyword in text for keyword in billing_keywords):
            return "ESCALATE"

    # -----------------------------------------------------
    # Subscription state
    # -----------------------------------------------------

    subscription_keywords = [
        "premium does not work",
        "still says free",
        "account has changed to free",
        "subscription",
        "premium has stopped working",
        "paying",
        "purchased premium",
    ]

    if intent == "PREMIUM_SUBSCRIPTION":
        if any(keyword in text for keyword in subscription_keywords):
            return "ESCALATE"

    # -----------------------------------------------------
    # Playlist / library account-specific problems
    # -----------------------------------------------------

    playlist_escalation_keywords = [
        "disappeared",
        "deleted",
        "lost",
        "don't sync",
        "doesn't sync",
        "different",
        "randomly deleting",
        "all my saved music",
    ]

    if intent == "PLAYLIST_LIBRARY":
        if any(
            keyword in text
            for keyword in playlist_escalation_keywords
        ):
            return "ESCALATE"

    # -----------------------------------------------------
    # General / vague messages
    # -----------------------------------------------------

    vague_phrases = [
        "no luck",
        "still the same",
        "help",
        "thanks",
        "thank you",
        "all clear",
        "not sure",
        "in my library",
        "please help",
    ]

    if intent == "GENERAL_OTHER":

        word_count = len(text.split())

        if word_count <= 8:
            return "ESCALATE"

        if any(
            phrase in text
            for phrase in vague_phrases
        ):
            return "ESCALATE"

    # -----------------------------------------------------
    # Everything else
    # -----------------------------------------------------

    return "AUTO_HANDLE"


# =========================================================
# MAIN
# =========================================================

def main():

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {GOLDEN_FILE}"
        )

    df = pd.read_csv(GOLDEN_FILE)

    # FIX: Replaced "expected_intent" and "expected_decision" with "label"
    required_columns = {
        "sample_id",
        "tweet_id",
        "text",
        "label",
    }   

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Add expected decision
    # -----------------------------------------------------

    df["expected_decision"] = df.apply(
        expected_decision,
        axis=1,
    )

    # -----------------------------------------------------
    # Prefer diverse examples
    #
    # Target:
    # 50 AUTO_HANDLE
    # 30 ESCALATE
    # -----------------------------------------------------

    auto = df[
        df["expected_decision"] == "AUTO_HANDLE"
    ].copy()

    escalate = df[
        df["expected_decision"] == "ESCALATE"
    ].copy()

    selected_parts = []

    # Take examples from every intent.
    # This prevents the evaluation set from being dominated
    # by GENERAL_OTHER / FEATURE_REQUEST.

    intents = sorted(df["label"].unique())

    for intent in intents:

        intent_auto = auto[
            auto["label"] == intent
        ].head(5)

        if len(intent_auto) > 0:
            selected_parts.append(intent_auto)

    # Combine initial AUTO_HANDLE candidates
    selected_auto = pd.concat(
        selected_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=["sample_id"]
    )

    # Fill remaining AUTO_HANDLE slots
    remaining_auto = auto[
        ~auto["sample_id"].isin(
            selected_auto["sample_id"]
        )
    ]

    selected_auto = pd.concat(
        [
            selected_auto,
            remaining_auto,
        ],
        ignore_index=True,
    ).head(50)

    # -----------------------------------------------------
    # Escalation examples
    # -----------------------------------------------------

    selected_escalate_parts = []

    for intent in intents:

        intent_escalate = escalate[
            escalate["label"] == intent
        ].head(4)

        if len(intent_escalate) > 0:
            selected_escalate_parts.append(
                intent_escalate
            )

    selected_escalate = pd.concat(
        selected_escalate_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=["sample_id"]
    )

    remaining_escalate = escalate[
        ~escalate["sample_id"].isin(
            selected_escalate["sample_id"]
        )
    ]

    selected_escalate = pd.concat(
        [
            selected_escalate,
            remaining_escalate,
        ],
        ignore_index=True,
    ).head(30)

    # -----------------------------------------------------
    # Final evaluation set
    # -----------------------------------------------------

    result = pd.concat(
        [
            selected_auto,
            selected_escalate,
        ],
        ignore_index=True,
    )

    result = result[
        [
            "sample_id",
            "tweet_id",
            "text",
            "label",
            "expected_decision",
        ]
    ].rename(
        columns={
            "label": "expected_intent",
        }
    )

    # Sort by sample ID
    result = result.sort_values(
        "sample_id"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("=" * 50)
    print("AGENT EVALUATION DATASET")
    print("=" * 50)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print(f"\nTotal examples: {len(result)}")

    print("\nDecision distribution:")
    print(
        result["expected_decision"]
        .value_counts()
        .to_string()
    )

    print("\nIntent distribution:")
    print(
        result["expected_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()