import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

JUDGE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "llm_judge_results.csv"
)

HUMAN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "human_reply_ratings.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "judge_human_agreement.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "judge_human_agreement_summary.csv"
)


# ============================================================
# RUBRIC DIMENSIONS
# ============================================================

DIMENSIONS = [
    "correctness",
    "groundedness",
    "helpfulness",
    "tone",
    "hallucination_free",
    "overall",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("       LLM JUDGE vs HUMAN AGREEMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(JUDGE_FILE):
        raise FileNotFoundError(
            f"Missing file:\n{JUDGE_FILE}"
        )

    if not os.path.exists(HUMAN_FILE):
        raise FileNotFoundError(
            f"Missing file:\n{HUMAN_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    judge_df = pd.read_csv(
        JUDGE_FILE
    )

    human_df = pd.read_csv(
        HUMAN_FILE
    )

    print(
        f"\nLLM judge rows: {len(judge_df)}"
    )

    print(
        f"Human rating rows: {len(human_df)}"
    )

    # --------------------------------------------------------
    # Normalize sample_id
    # --------------------------------------------------------

    judge_df["sample_id"] = (
        judge_df["sample_id"]
        .astype(str)
        .str.strip()
    )

    human_df["sample_id"] = (
        human_df["sample_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Keep only rows with complete LLM scores
    # --------------------------------------------------------

    judge_df = judge_df.dropna(
        subset=DIMENSIONS
    ).copy()

    print(
        f"Valid LLM judge rows: "
        f"{len(judge_df)}"
    )

    # --------------------------------------------------------
    # Convert human scores to numeric
    # --------------------------------------------------------

    for dimension in DIMENSIONS:

        human_df[dimension] = pd.to_numeric(
            human_df[dimension],
            errors="coerce"
        )

    # Keep only complete human ratings.
    human_df = human_df.dropna(
        subset=DIMENSIONS
    ).copy()

    print(
        f"Valid human rating rows: "
        f"{len(human_df)}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = judge_df.merge(
        human_df,
        on="sample_id",
        suffixes=(
            "_llm",
            "_human"
        ),
        how="inner"
    )

    print(
        f"Matched rows: {len(merged)}"
    )

    if len(merged) < 2:

        raise ValueError(
            "Not enough overlapping rows with valid "
            "LLM and human scores."
        )

    # --------------------------------------------------------
    # Show matched IDs
    # --------------------------------------------------------

    print("\nMatched sample IDs:")

    print(
        ", ".join(
            merged["sample_id"]
            .tolist()
        )
    )

    # ========================================================
    # BUILD DETAILED COMPARISON
    # ========================================================

    comparison_rows = []

    for _, row in merged.iterrows():

        result = {
            "sample_id":
                row["sample_id"]
        }

        for dimension in DIMENSIONS:

            llm_score = float(
                row[
                    f"{dimension}_llm"
                ]
            )

            human_score = float(
                row[
                    f"{dimension}_human"
                    ]
            )

            result[
                f"{dimension}_llm"
            ] = llm_score

            result[
                f"{dimension}_human"
            ] = human_score

            result[
                f"{dimension}_difference"
            ] = (
                llm_score
                - human_score
            )

        comparison_rows.append(
            result
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # AGREEMENT STATISTICS
    # ========================================================

    print("\n" + "-" * 60)
    print("AGREEMENT RESULTS")
    print("-" * 60)

    summary_rows = []

    for dimension in DIMENSIONS:

        llm = pd.to_numeric(
            merged[
                f"{dimension}_llm"
            ],
            errors="coerce"
        )

        human = pd.to_numeric(
            merged[
                f"{dimension}_human"
            ],
            errors="coerce"
        )

        mask = (
            llm.notna()
            & human.notna()
        )

        llm = llm[mask].astype(float)
        human = human[mask].astype(float)

        if len(llm) < 2:
            continue

        # ----------------------------------------------------
        # Pearson
        # ----------------------------------------------------

        try:

            pearson_r, pearson_p = pearsonr(
                llm,
                human
            )

        except Exception:

            pearson_r = np.nan
            pearson_p = np.nan

        # ----------------------------------------------------
        # Spearman
        # ----------------------------------------------------

        try:

            spearman_r, spearman_p = spearmanr(
                llm,
                human
            )

        except Exception:

            spearman_r = np.nan
            spearman_p = np.nan

        # ----------------------------------------------------
        # Mean absolute error
        # ----------------------------------------------------

        mae = np.mean(
            np.abs(
                llm.to_numpy()
                - human.to_numpy()
            )
        )

        # ----------------------------------------------------
        # Exact agreement
        # ----------------------------------------------------

        exact_agreement = np.mean(
            llm.to_numpy()
            == human.to_numpy()
        )

        # ----------------------------------------------------
        # Near agreement: within 1 point
        # ----------------------------------------------------

        near_agreement = np.mean(
            np.abs(
                llm.to_numpy()
                - human.to_numpy()
            )
            <= 1
        )

        summary_rows.append(
            {
                "dimension":
                    dimension,

                "n":
                    len(llm),

                "pearson_r":
                    pearson_r,

                "pearson_p":
                    pearson_p,

                "spearman_r":
                    spearman_r,

                "spearman_p":
                    spearman_p,

                "mean_absolute_difference":
                    mae,

                "exact_agreement":
                    exact_agreement,

                "within_1_agreement":
                    near_agreement,
            }
        )

        print(
            f"\n{dimension.upper()}"
        )

        print(
            f"  N:                    {len(llm)}"
        )

        print(
            f"  Pearson r:            "
            f"{pearson_r:.3f}"
        )

        print(
            f"  Spearman rho:         "
            f"{spearman_r:.3f}"
        )

        print(
            f"  Mean abs difference:  "
            f"{mae:.3f}"
        )

        print(
            f"  Exact agreement:      "
            f"{exact_agreement * 100:.1f}%"
        )

        print(
            f"  Within ±1 agreement:  "
            f"{near_agreement * 100:.1f}%"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n" + "-" * 60)
    print("AVERAGE ACROSS DIMENSIONS")
    print("-" * 60)

    mean_pearson = summary_df[
        "pearson_r"
    ].mean()

    mean_spearman = summary_df[
        "spearman_r"
    ].mean()

    mean_mae = summary_df[
        "mean_absolute_difference"
    ].mean()

    mean_exact = summary_df[
        "exact_agreement"
    ].mean()

    mean_within_1 = summary_df[
        "within_1_agreement"
    ].mean()

    print(
        f"Mean Pearson r:           "
        f"{mean_pearson:.3f}"
    )

    print(
        f"Mean Spearman rho:        "
        f"{mean_spearman:.3f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_mae:.3f}"
    )

    print(
        f"Mean exact agreement:     "
        f"{mean_exact * 100:.1f}%"
    )

    print(
        f"Mean within ±1:           "
        f"{mean_within_1 * 100:.1f}%"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("              COMPLETE")
    print("=" * 60)

    print(
        f"\nDetailed comparison:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()