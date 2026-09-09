import os
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPLIES_FILE = os.path.join(PROJECT_ROOT, "data", "reply_evaluation.csv")
JUDGE_FILE = os.path.join(PROJECT_ROOT, "data", "llm_judge_results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "human_reply_ratings.csv")

def main():
    print("=" * 60)
    print("       CREATE HUMAN RATING TEMPLATE")
    print("=" * 60)

    replies = pd.read_csv(REPLIES_FILE)
    judge = pd.read_csv(JUDGE_FILE)

    score_columns = [
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall",
    ]

    # Keep only rows where ALL LLM judge scores exist.
    valid_judge = judge.dropna(subset=score_columns).copy()
    print(f"Valid LLM-judged examples: {len(valid_judge)}")

    merged = replies.merge(valid_judge[["sample_id"]], on="sample_id", how="inner")

    # Select the first 25 successfully judged examples.
    merged = merged.head(25).copy()
    print(f"Human-rating examples created: {len(merged)}")

    # Create blank human-rating columns.
    output = merged[[
        "sample_id",
        "text",
        "intent",
        "historical_evidence",
        "generated_reply",
    ]].copy()

    output["correctness"] = ""
    output["groundedness"] = ""
    output["helpfulness"] = ""
    output["tone"] = ""
    output["hallucination_free"] = ""
    output["overall"] = ""
    output["human_reason"] = ""

    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nCreated:\n{OUTPUT_FILE}")
    print("\nColumns:")
    for col in output.columns:
        print(f" - {col}")

if __name__ == "__main__":
    main()