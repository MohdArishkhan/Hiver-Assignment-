import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# Allow imports from scripts/
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from support_agent import (  # noqa: E402
    predict_intent,
    retrieve_cases,
    decide_escalation,
)


INPUT_FILE = ROOT / "data" / "agent_eval.csv"
OUTPUT_FILE = ROOT / "data" / "agent_evaluation_results.csv"


# =========================================================
# EVALUATE ONE EXAMPLE
# =========================================================

def evaluate_example(text: str) -> dict:
    """
    Run only the deterministic portion of the existing agent.

    No LLM call is made here.
    """

    intent = predict_intent(text)

    retrieved_cases = retrieve_cases(text)

    decision, reason = decide_escalation(
        text,
        intent,
        retrieved_cases,
    )

    retrieval_similarity = ""

    if retrieved_cases:
        retrieval_similarity = retrieved_cases[0]["similarity"]

    return {
        "predicted_intent": intent,
        "predicted_decision": decision,
        "retrieval_similarity": retrieval_similarity,
        "reason": reason,
    }


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(df: pd.DataFrame) -> None:

    valid = df[
        (df["predicted_intent"] != "")
        & (df["predicted_decision"] != "")
    ].copy()

    if valid.empty:
        print("\nNo valid predictions found.")
        return

    # -----------------------------------------------------
    # Intent
    # -----------------------------------------------------

    intent_accuracy = accuracy_score(
        valid["expected_intent"],
        valid["predicted_intent"],
    )

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    decision_accuracy = accuracy_score(
        valid["expected_decision"],
        valid["predicted_decision"],
    )

    (
        escalation_precision,
        escalation_recall,
        escalation_f1,
        _,
    ) = precision_recall_fscore_support(
        valid["expected_decision"],
        valid["predicted_decision"],
        labels=["ESCALATE"],
        average="binary",
        pos_label="ESCALATE",
        zero_division=0,
    )

    # -----------------------------------------------------
    # False Auto-Handle
    #
    # Dangerous error:
    # expected ESCALATE
    # predicted AUTO_HANDLE
    # -----------------------------------------------------

    false_auto_handle = (
        (valid["expected_decision"] == "ESCALATE")
        & (valid["predicted_decision"] == "AUTO_HANDLE")
    ).sum()

    expected_escalations = (
        valid["expected_decision"] == "ESCALATE"
    ).sum()

    false_auto_handle_rate = (
        false_auto_handle / expected_escalations
        if expected_escalations > 0
        else 0.0
    )

    # -----------------------------------------------------
    # False Escalation
    #
    # Expected AUTO_HANDLE
    # Predicted ESCALATE
    # -----------------------------------------------------

    false_escalation = (
        (valid["expected_decision"] == "AUTO_HANDLE")
        & (valid["predicted_decision"] == "ESCALATE")
    ).sum()

    expected_auto_handles = (
        valid["expected_decision"] == "AUTO_HANDLE"
    ).sum()

    false_escalation_rate = (
        false_escalation / expected_auto_handles
        if expected_auto_handles > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("        SPOTIFY AGENT EVALUATION")
    print("=" * 50)

    print(f"\nExamples evaluated: {len(valid)}")

    print("\nIntent metrics")
    print("-" * 30)
    print(f"Intent Accuracy: {intent_accuracy:.4f}")

    print("\nDecision metrics")
    print("-" * 30)
    print(f"Decision Accuracy:      {decision_accuracy:.4f}")
    print(f"Escalation Precision:   {escalation_precision:.4f}")
    print(f"Escalation Recall:      {escalation_recall:.4f}")
    print(f"Escalation F1:          {escalation_f1:.4f}")

    print("\nSafety metrics")
    print("-" * 30)
    print(
        f"False Auto-Handle Rate: "
        f"{false_auto_handle_rate:.4f}"
    )
    print(
        f"False Escalation Rate:  "
        f"{false_escalation_rate:.4f}"
    )

    # -----------------------------------------------------
    # Counts
    # -----------------------------------------------------

    print("\nDecision counts")
    print("-" * 30)

    print(
        "Expected ESCALATE:",
        expected_escalations,
    )

    print(
        "Expected AUTO_HANDLE:",
        expected_auto_handles,
    )

    print(
        "False AUTO_HANDLE:",
        false_auto_handle,
    )

    print(
        "False ESCALATION:",
        false_escalation,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Spotify support agent without LLM calls."
    )

    parser.add_argument(
        "--input",
        default=str(INPUT_FILE),
        help="Path to agent evaluation CSV",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help="Path to save evaluation results",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # -----------------------------------------------------
    # Validate files
    # -----------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {input_path}"
        )

    # -----------------------------------------------------
    # Load evaluation data
    # -----------------------------------------------------

    df = pd.read_csv(input_path)

    required_columns = {
        "sample_id",
        "text",
        "expected_intent",
        "expected_decision",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "agent_eval.csv is empty. "
            "Add evaluation examples before running the evaluator."
        )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = []

    total = len(df)

    for index, row in df.iterrows():

        sample_id = row["sample_id"]
        text = str(row["text"]).strip()

        print(
            f"[{index + 1}/{total}] "
            f"Evaluating sample {sample_id}..."
        )

        try:

            prediction = evaluate_example(text)

            results.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "expected_intent": row["expected_intent"],
                    "predicted_intent": prediction[
                        "predicted_intent"
                    ],
                    "expected_decision": row[
                        "expected_decision"
                    ],
                    "predicted_decision": prediction[
                        "predicted_decision"
                    ],
                    "retrieval_similarity": prediction[
                        "retrieval_similarity"
                    ],
                    "reply": "",
                    "reason": prediction["reason"],
                    "error": "",
                }
            )

        except Exception as exc:

            results.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "expected_intent": row["expected_intent"],
                    "predicted_intent": "",
                    "expected_decision": row[
                        "expected_decision"
                    ],
                    "predicted_decision": "",
                    "retrieval_similarity": "",
                    "reply": "",
                    "reason": "",
                    "error": str(exc),
                }
            )

            print(
                f"  ERROR: {exc}"
            )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    print()
    print("Results saved to:")
    print(output_path)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    calculate_metrics(results_df)


if __name__ == "__main__":
    main()