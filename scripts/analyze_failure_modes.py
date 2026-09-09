import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AGENT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "agent_eval.csv"
)

REPLY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "reply_evaluation.csv"
)

JUDGE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "llm_judge_results.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "failure_analysis.csv"
)


# ============================================================
# HELPERS
# ============================================================

def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def classify_failure(row):
    """
    Assign one primary failure mode using deterministic rules.

    Priority:
    1. Very poor generated reply
    2. Wrong intent
    3. Weak retrieval
    4. Generic / incomplete response
    5. Other reply-quality issue
    """

    overall = row.get("overall")
    helpfulness = row.get("helpfulness")
    groundedness = row.get("groundedness")
    retrieval = row.get("retrieval_similarity")

    expected_intent = row.get("expected_intent")
    predicted_intent = row.get("intent")

    # --------------------------------------------------------
    # 1. Intent mismatch
    # --------------------------------------------------------

    if (
        pd.notna(expected_intent)
        and pd.notna(predicted_intent)
        and str(expected_intent).strip()
        != str(predicted_intent).strip()
    ):
        return "Wrong intent prediction"

    # --------------------------------------------------------
    # 2. Very poor reply
    # --------------------------------------------------------

    if pd.notna(overall) and overall <= 2:
        return "Poor overall reply quality"

    # --------------------------------------------------------
    # 3. Weak retrieval
    # --------------------------------------------------------

    if pd.notna(retrieval) and retrieval < 0.25:
        return "Weak historical retrieval"

    # --------------------------------------------------------
    # 4. Generic / incomplete reply
    # --------------------------------------------------------

    if (
        pd.notna(helpfulness)
        and helpfulness <= 2
    ):
        return "Generic or insufficiently helpful reply"

    # --------------------------------------------------------
    # 5. Grounding problem
    # --------------------------------------------------------

    if (
        pd.notna(groundedness)
        and groundedness <= 2
    ):
        return "Weak grounding"

    return "Other"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("                 FINAL FAILURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    agent_df = pd.read_csv(
        AGENT_FILE
    )

    reply_df = pd.read_csv(
        REPLY_FILE
    )

    judge_df = pd.read_csv(
        JUDGE_FILE
    )

    print(
        f"\nAgent evaluation rows: {len(agent_df)}"
    )

    print(
        f"Reply evaluation rows: {len(reply_df)}"
    )

    print(
        f"LLM judge rows: {len(judge_df)}"
    )

    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    for df in [
        agent_df,
        reply_df,
        judge_df
    ]:
        df["sample_id"] = (
            df["sample_id"]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Merge reply + agent data
    # --------------------------------------------------------

    merged = reply_df.merge(
        agent_df,
        on="sample_id",
        how="left",
        suffixes=(
            "",
            "_agent"
        )
    )

    # --------------------------------------------------------
    # Merge judge results
    # --------------------------------------------------------

    merged = merged.merge(
        judge_df,
        on="sample_id",
        how="left",
        suffixes=(
            "",
            "_judge"
        )
    )

    # --------------------------------------------------------
    # Resolve expected intent column
    # --------------------------------------------------------

    expected_intent_col = first_existing_column(
        merged,
        [
            "expected_intent",
            "gold_intent",
            "true_intent",
            "label",
            "expected_label"
        ]
    )

    if expected_intent_col:
        merged["expected_intent"] = (
            merged[expected_intent_col]
        )
    else:
        merged["expected_intent"] = pd.NA

    # --------------------------------------------------------
    # Retrieval similarity numeric
    # --------------------------------------------------------

    merged["retrieval_similarity"] = pd.to_numeric(
        merged["retrieval_similarity"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Judge scores numeric
    # --------------------------------------------------------

    for col in [
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall"
    ]:

        if col in merged.columns:
            merged[col] = pd.to_numeric(
                merged[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Only use rows that actually have judge results
    # --------------------------------------------------------

    judged = merged[
        merged["overall"].notna()
    ].copy()

    print(
        f"\nRows with successful LLM judge score: "
        f"{len(judged)}"
    )

    if len(judged) == 0:
        raise ValueError(
            "No successfully judged rows available."
        )

    # --------------------------------------------------------
    # Assign failure modes
    # --------------------------------------------------------

    judged["failure_mode"] = judged.apply(
        classify_failure,
        axis=1
    )

    # --------------------------------------------------------
    # Rank severity
    # --------------------------------------------------------

    judged["severity"] = (
        5 - judged["overall"]
    )

    judged = judged.sort_values(
        [
            "severity",
            "retrieval_similarity"
        ],
        ascending=[
            False,
            True
        ]
    )

    # --------------------------------------------------------
    # Save detailed table
    # --------------------------------------------------------

    output_columns = [
        "sample_id",
        "text",
        "expected_intent",
        "intent",
        "retrieval_similarity",
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall",
        "failure_mode",
        "generated_reply",
        "judge_reason"
    ]

    # Keep only columns that exist.
    output_columns = [
        col
        for col in output_columns
        if col in judged.columns
    ]

    failure_df = judged[
        output_columns
    ].copy()

    failure_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # TOP 5 INDIVIDUAL FAILURE EXAMPLES
    # ========================================================

    print("\n" + "-" * 70)
    print("TOP 5 FAILURE EXAMPLES")
    print("-" * 70)

    top5 = judged.head(5)

    for rank, (_, row) in enumerate(
        top5.iterrows(),
        start=1
    ):

        print(
            f"\n#{rank} "
            f"Sample ID: {row['sample_id']}"
        )

        print(
            f"Failure mode: "
            f"{row['failure_mode']}"
        )

        print(
            f"Overall score: "
            f"{row['overall']}/5"
        )

        print(
            f"Retrieval similarity: "
            f"{row['retrieval_similarity']:.4f}"
        )

        print(
            f"Intent: "
            f"{row['intent']}"
        )

        if pd.notna(
            row.get("expected_intent")
        ):
            print(
                f"Expected intent: "
                f"{row['expected_intent']}"
            )

        print(
            "\nCustomer:"
        )

        print(
            str(row["text"])
        )

        print(
            "\nGenerated reply:"
        )

        print(
            str(row["generated_reply"])
        )

        print(
            "\nJudge reason:"
        )

        print(
            str(row["judge_reason"])
        )

    # ========================================================
    # FAILURE MODE COUNTS
    # ========================================================

    print("\n" + "-" * 70)
    print("FAILURE MODE COUNTS")
    print("-" * 70)

    counts = (
        judged["failure_mode"]
        .value_counts()
    )

    for mode, count in counts.items():

        print(
            f"{mode:45s} {count}"
        )

    # ========================================================
    # LOWEST-SCORING REPLIES
    # ========================================================

    print("\n" + "-" * 70)
    print("LOWEST-SCORING REPLIES")
    print("-" * 70)

    lowest = judged.sort_values(
        "overall"
    ).head(10)

    print(
        lowest[
            [
                "sample_id",
                "overall",
                "correctness",
                "groundedness",
                "helpfulness",
                "retrieval_similarity",
                "failure_mode"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("                   ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    print(
        "\nThe generated CSV can be used as the reproducible "
        "failure-analysis evidence in the final report."
    )


if __name__ == "__main__":
    main()