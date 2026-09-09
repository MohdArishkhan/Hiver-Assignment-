import os
import sys
import pandas as pd
import joblib

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORT EXISTING FUNCTIONS
# ============================================================

from scripts.retrieve_cases import retrieve_cases
from scripts.generate_reply import generate_reply

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "agent_eval.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "reply_evaluation.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "final_model.pkl")

# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 3

# ============================================================
# LOAD FINAL CLASSIFIER ONCE
# ============================================================

print("Loading intent classifier...")
model = joblib.load(MODEL_FILE)

# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(text):
    """
    Predict the customer's intent using the saved final model.
    """
    prediction = model.predict([text])[0]
    return prediction

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("              REPLY EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD AGENT EVALUATION DATA
    # --------------------------------------------------------
    print("\nLoading agent evaluation data...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find evaluation file:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"Total benchmark examples: {len(df)}")

    # --------------------------------------------------------
    # 2. SELECT AUTO_HANDLE CASES ONLY
    # --------------------------------------------------------
    if "expected_decision" in df.columns:
        auto_df = df[
            df["expected_decision"].astype(str).str.upper().str.strip() == "AUTO_HANDLE"
        ].copy()
    elif "decision" in df.columns:
        auto_df = df[
            df["decision"].astype(str).str.upper().str.strip() == "AUTO_HANDLE"
        ].copy()
    else:
        raise ValueError(
            "agent_eval.csv must contain either 'expected_decision' or 'decision'."
        )

    print(f"AUTO_HANDLE examples: {len(auto_df)}")

    if len(auto_df) == 0:
        raise ValueError("No AUTO_HANDLE examples found.")

    # --------------------------------------------------------
    # 3. GENERATE REPLIES
    # --------------------------------------------------------
    results = []

    for i, (_, row) in enumerate(auto_df.iterrows(), start=1):
        sample_id = row.get("sample_id")
        text = str(row.get("text", "")).strip()
        tweet_id = row.get("tweet_id")

        print("\n" + "-" * 60)
        print(f"[{i}/{len(auto_df)}] Sample ID: {sample_id}")
        print("\nCustomer:")
        print(text)

        # Skip empty messages
        if not text:
            print("\nSkipping empty customer message.")
            continue

        try:
            # STEP 1 — INTENT CLASSIFICATION
            intent = predict_intent(text)
            print("\nPredicted intent:")
            print(intent)

            # STEP 2 — HISTORICAL RETRIEVAL
            if pd.notna(tweet_id):
                exclude_ids = [tweet_id]
            else:
                exclude_ids = None

            tweet_id = row.get("tweet_id")

            evidence = retrieve_cases(
                text,
                top_k=TOP_K,
                exclude_tweet_ids=(
                    [tweet_id]
                    if pd.notna(tweet_id)
                    else None
                )
            )

            # Best retrieval similarity
            if evidence:
                retrieval_similarity = float(evidence[0].get("similarity", 0.0))
            else:
                retrieval_similarity = 0.0

            print("\nBest retrieval similarity:", round(retrieval_similarity, 4))
            print(f"Retrieved historical cases: {len(evidence)}")

            # STEP 3 — GROUNDED REPLY GENERATION
            generated_reply = generate_reply(
                customer_message=text,
                intent=intent,
                evidence=evidence
            )

            print("\nGenerated reply:")
            print(generated_reply)

            # STEP 4 — SAVE RESULT
            results.append({
                "sample_id": sample_id,
                "text": text,
                "intent": intent,
                "retrieval_similarity": retrieval_similarity,
                "historical_evidence": str(evidence),
                "generated_reply": generated_reply
            })

        except Exception as e:
            print(f"\nERROR processing sample {sample_id}:")
            print(e)
            results.append({
                "sample_id": sample_id,
                "text": text,
                "intent": "",
                "retrieval_similarity": 0.0,
                "historical_evidence": "",
                "generated_reply": ""
            })

    # --------------------------------------------------------
    # 4. SAVE RESULTS
    # --------------------------------------------------------
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # --------------------------------------------------------
    # 5. FINAL SUMMARY
    # --------------------------------------------------------
    successful_replies = (
        output_df["generated_reply"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    failed_replies = len(output_df) - successful_replies

    print("\n" + "=" * 60)
    print("           EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nInput examples:       {len(auto_df)}")
    print(f"Rows saved:           {len(output_df)}")
    print(f"Successful replies:   {successful_replies}")
    print(f"Failed/empty replies: {failed_replies}")
    print(f"\nOutput file:\n{OUTPUT_FILE}")
    print("\nColumns:")
    for column in output_df.columns:
        print(f" - {column}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()