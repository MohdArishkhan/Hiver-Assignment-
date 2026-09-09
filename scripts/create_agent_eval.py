from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GOLDEN_FILE = ROOT / "data" / "golden_200.csv"
OUTPUT_FILE = ROOT / "data" / "agent_eval.csv"


# =========================================================
# EXPECTED DECISION POLICY
# =========================================================

def expected_decision(row):

    text = str(row["text"]).lower()
    intent = str(row["label"])

    # -----------------------------------------------------
    # Security risk
    # -----------------------------------------------------

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

    if any(signal in text for signal in security_signals):
        return "ESCALATE"

    # -----------------------------------------------------
    # Billing / payment events
    # -----------------------------------------------------

    billing_signals = [
        "charged",
        "charging",
        "charged twice",
        "duplicate charge",
        "wrong charge",
        "unexpected charge",
        "fraudulent charge",
        "refund",
        "billing issue",
        "payment failed",
        "payment won't go through",
        "payment went through",
        "money taken",
        "money was taken",
        "bank statement",
        "charged my card",
        "charging my account",
        "payment details",
        "payment information",
        "update my payment",
        "update payment",
        "cancelled",
        "canceled",
    ]

    if any(signal in text for signal in billing_signals):
        return "ESCALATE"

    # -----------------------------------------------------
    # Premium account-state problems
    # -----------------------------------------------------

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
        "don't have premium",
        "do not have premium",
        "dont have premium",
        "no premium",
    ]

    if any(signal in text for signal in premium_state_signals):
        return "ESCALATE"

    # -----------------------------------------------------
    # Playlist / library recovery
    # -----------------------------------------------------

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
        "restore my playlists",
        "restore previous playlist",
        "restore previous week's playlist",
        "lost my favourite song",
        "lost my favorite song",
        "deleted my entire library",
        "deleting my entire library",
        "entire library",
        "music disappeared",
        "music disappear",
        "all my music",
    ]

    if any(signal in text for signal in library_signals):
        return "ESCALATE"

    # -----------------------------------------------------
    # Vague unresolved requests
    # -----------------------------------------------------

    vague_signals = [
        "still the same",
        "no luck",
        "same issue",
        "not sure",
        "no solution yet",
    ]

    if len(text.split()) <= 8:
        if any(signal in text for signal in vague_signals):
            return "ESCALATE"

    # -----------------------------------------------------
    # Default
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
    # Calculate expected decision
    # -----------------------------------------------------

    df["expected_decision"] = df.apply(
        expected_decision,
        axis=1,
    )

    # -----------------------------------------------------
    # Manual corrections
    # -----------------------------------------------------

    # General informational Premium question.
    df.loc[
        df["sample_id"] == 6,
        "expected_decision"
    ] = "AUTO_HANDLE"

    # Payment-method question, not a payment dispute.
    df.loc[
        df["sample_id"] == 90,
        "expected_decision"
    ] = "AUTO_HANDLE"

    # -----------------------------------------------------
    # Remove non-actionable acknowledgements
    # -----------------------------------------------------

    excluded_ids = {
        26,
        176,
        251,
        252,
        412,
    }

    df = df[
        ~df["sample_id"].isin(excluded_ids)
    ].copy()

    # -----------------------------------------------------
    # Split candidates
    # -----------------------------------------------------

    auto = df[
        df["expected_decision"] == "AUTO_HANDLE"
    ].copy()

    escalate = df[
        df["expected_decision"] == "ESCALATE"
    ].copy()

    print("\nCandidates")
    print("------------------------------")
    print("AUTO_HANDLE:", len(auto))
    print("ESCALATE:   ", len(escalate))

    # -----------------------------------------------------
    # Need enough examples
    # -----------------------------------------------------

    if len(auto) < 50:
        raise ValueError(
            f"Not enough AUTO_HANDLE candidates: {len(auto)}"
        )

    if len(escalate) < 25:
        raise ValueError(
            f"Not enough ESCALATE candidates: {len(escalate)}"
        )

    # -----------------------------------------------------
    # Select 50 AUTO_HANDLE
    #
    # Prefer balanced intent coverage.
    # -----------------------------------------------------

    auto_parts = []

    for intent in sorted(auto["label"].unique()):

        rows = auto[
            auto["label"] == intent
        ].head(4)

        auto_parts.append(rows)

    selected_auto = pd.concat(
        auto_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=["sample_id"]
    )

    # Fill remaining slots.
    if len(selected_auto) < 50:

        remaining = auto[
            ~auto["sample_id"].isin(
                selected_auto["sample_id"]
            )
        ]

        selected_auto = pd.concat(
            [
                selected_auto,
                remaining,
            ],
            ignore_index=True,
        )

    selected_auto = selected_auto.head(50)

    # -----------------------------------------------------
    # Select 25 ESCALATE
    # -----------------------------------------------------

    escalate_parts = []

    for intent in sorted(escalate["label"].unique()):

        rows = escalate[
            escalate["label"] == intent
        ].head(3)

        escalate_parts.append(rows)

    selected_escalate = pd.concat(
        escalate_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=["sample_id"]
    )

    if len(selected_escalate) < 25:

        remaining = escalate[
            ~escalate["sample_id"].isin(
                selected_escalate["sample_id"]
            )
        ]

        selected_escalate = pd.concat(
            [
                selected_escalate,
                remaining,
            ],
            ignore_index=True,
        )

    selected_escalate = selected_escalate.head(25)

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    result = pd.concat(
        [
            selected_auto,
            selected_escalate,
        ],
        ignore_index=True,
    )

    # Keep only required columns.
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

    result = result.sort_values(
        "sample_id"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # IMPORTANT VALIDATION
    # -----------------------------------------------------

    if len(result) != 75:
        raise ValueError(
            f"Expected 75 rows, got {len(result)}"
        )

    counts = (
        result["expected_decision"]
        .value_counts()
        .to_dict()
    )

    auto_count = counts.get(
        "AUTO_HANDLE",
        0,
    )

    escalate_count = counts.get(
        "ESCALATE",
        0,
    )

    if auto_count != 50:
        raise ValueError(
            f"Expected 50 AUTO_HANDLE, got {auto_count}"
        )

    if escalate_count != 25:
        raise ValueError(
            f"Expected 25 ESCALATE, got {escalate_count}"
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
    # RELOAD THE FILE
    #
    # This catches save/output inconsistencies.
    # -----------------------------------------------------

    saved = pd.read_csv(
        OUTPUT_FILE
    )

    saved_counts = (
        saved["expected_decision"]
        .value_counts()
        .to_dict()
    )

    saved_auto = saved_counts.get(
        "AUTO_HANDLE",
        0,
    )

    saved_escalate = saved_counts.get(
        "ESCALATE",
        0,
    )

    if len(saved) != 75:
        raise ValueError(
            f"Saved CSV has {len(saved)} rows, expected 75"
        )

    if saved_auto != 50:
        raise ValueError(
            f"Saved CSV has {saved_auto} AUTO_HANDLE rows, expected 50"
        )

    if saved_escalate != 25:
        raise ValueError(
            f"Saved CSV has {saved_escalate} ESCALATE rows, expected 25"
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("AGENT EVALUATION DATASET")
    print("=" * 50)

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nTotal examples:", len(saved))

    print("\nDecision distribution:")
    print(
        saved["expected_decision"]
        .value_counts()
        .to_string()
    )

    print("\nIntent distribution:")
    print(
        saved["expected_intent"]
        .value_counts()
        .to_string()
    )

    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()