import os
import json
import re
import time
import pandas as pd
from openai import OpenAI


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "reply_evaluation.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "llm_judge_results.csv"
)


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

MODEL = "google/gemini-2.5-flash"

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set.\n"
        "PowerShell:\n"
        "$env:OPENROUTER_API_KEY='your-key-here'"
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer":
            "https://github.com/MohdArishkhan/Hiver-Assignment-",
        "X-Title":
            "Spotify Support Agent - LLM Judge",
    },
)


# ============================================================
# RUBRIC
# ============================================================

RUBRIC = """
Score the customer-support reply on each dimension from 1 to 5.

CORRECTNESS
5 = Correctly addresses the customer's issue and does not make materially false claims.
4 = Mostly correct with a minor omission or imprecision.
3 = Partially correct; useful but misses an important part.
2 = Mostly incorrect, poorly targeted, or materially incomplete.
1 = Incorrect or clearly fails to address the issue.

GROUNDEDNESS
5 = Fully supported by the supplied historical Spotify evidence and customer message.
4 = Mostly grounded, with only a small amount of reasonable paraphrasing/inference.
3 = Some grounding, but includes notable unsupported content.
2 = Largely unsupported by the evidence.
1 = Invents policies, actions, guarantees, facts, or other unsupported information.

HELPFULNESS
5 = Gives a clear, relevant next step or useful answer for the customer.
4 = Helpful but could be more specific or actionable.
3 = Moderately helpful but incomplete or generic.
2 = Barely useful.
1 = Not helpful or does not meaningfully address the customer.

TONE
5 = Professional, concise, empathetic, and natural for customer support.
4 = Good tone with a minor stylistic issue.
3 = Acceptable but noticeably generic, awkward, or repetitive.
2 = Poor tone, overly dismissive, confusing, or inappropriate.
1 = Clearly unprofessional or inappropriate.

HALLUCINATION_FREE
5 = No unsupported claims, invented URLs, invented actions, guarantees, or facts.
4 = Very minor questionable inference that does not materially mislead.
3 = Contains some unsupported information.
2 = Contains a significant unsupported claim.
1 = Contains clear hallucinations, invented actions/policies/URLs, or misleading claims.

OVERALL
5 = Strong production-quality support reply.
4 = Good reply with minor issues.
3 = Usable but has meaningful weaknesses.
2 = Poor quality and would need substantial revision.
1 = Unsafe, misleading, or unusable.
"""


# ============================================================
# HELPERS
# ============================================================

def clean_json_text(text):
    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def extract_json(text):
    cleaned = clean_json_text(text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL
    )

    if match:
        return json.loads(match.group(0))

    raise ValueError(
        "Could not parse judge response as JSON."
    )


def validate_score(value, field):
    try:
        value = int(value)

    except (TypeError, ValueError):
        raise ValueError(
            f"{field} is not an integer: {value}"
        )

    if value < 1 or value > 5:
        raise ValueError(
            f"{field} must be between 1 and 5: {value}"
        )

    return value


def is_valid_judgment(row):
    """
    A row is considered successfully judged only if
    all six rubric scores are present.
    """

    score_columns = [
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall",
    ]

    for column in score_columns:

        value = row.get(column)

        if pd.isna(value):
            return False

        try:
            value = int(value)

        except (TypeError, ValueError):
            return False

        if value < 1 or value > 5:
            return False

    return True


# ============================================================
# JUDGE ONE REPLY
# ============================================================

def judge_reply(
    customer_message,
    predicted_intent,
    historical_evidence,
    generated_reply
):

    prompt = f"""
You are evaluating an AI-generated Spotify customer-support reply.

Use ONLY the information provided below.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent}

HISTORICAL SPOTIFY SUPPORT EVIDENCE:
{historical_evidence}

GENERATED REPLY:
{generated_reply}

RUBRIC:
{RUBRIC}

IMPORTANT:
- Judge the generated reply, not the classifier itself.
- Do not assume facts that are not present in the customer message
  or historical evidence.
- Penalize invented Spotify policies, unsupported account actions,
  unsupported guarantees, and invented URLs.
- A concise reply can score highly if it is appropriately helpful.
- Asking for more information is acceptable when the evidence is insufficient.
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown.

Return exactly this schema:

{{
  "correctness": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "tone": 1,
  "hallucination_free": 1,
  "overall": 1,
  "judge_reason": "Brief explanation of the scores."
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of customer-support "
                    "reply quality. Be consistent and evidence-based."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=500,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError(
            "LLM judge returned an empty response."
        )

    result = extract_json(content)

    required_fields = [
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall",
        "judge_reason",
    ]

    for field in required_fields:

        if field not in result:
            raise ValueError(
                f"Missing field from judge response: {field}"
            )

    for field in required_fields[:-1]:

        result[field] = validate_score(
            result[field],
            field
        )

    result["judge_reason"] = str(
        result["judge_reason"]
    ).strip()

    return result


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if not os.path.exists(OUTPUT_FILE):
        return []

    try:

        existing_df = pd.read_csv(
            OUTPUT_FILE
        )

    except Exception:

        return []

    if existing_df.empty:
        return []

    valid_rows = []

    for _, row in existing_df.iterrows():

        if is_valid_judgment(row):

            valid_rows.append(
                row.to_dict()
            )

    return valid_rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("          RESUME-SAFE LLM REPLY JUDGE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}\n\n"
            "Run evaluate_replies.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nTotal replies available: {len(df)}"
    )

    # --------------------------------------------------------
    # Load previous successful judgments
    # --------------------------------------------------------

    results = load_existing_results()

    judged_ids = {
        str(row["sample_id"])
        for row in results
    }

    print(
        f"Already completed: {len(judged_ids)}"
    )

    remaining_df = df[
        ~df["sample_id"]
        .astype(str)
        .isin(judged_ids)
    ].copy()

    print(
        f"Remaining: {len(remaining_df)}"
    )

    if len(remaining_df) == 0:

        print(
            "\nAll replies are already judged."
        )

        return

    # --------------------------------------------------------
    # Judge remaining cases
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        remaining_df.iterrows(),
        start=1
    ):

        sample_id = row.get(
            "sample_id"
        )

        customer_message = str(
            row.get(
                "text",
                ""
            )
        ).strip()

        predicted_intent = str(
            row.get(
                "intent",
                ""
            )
        ).strip()

        historical_evidence = str(
            row.get(
                "historical_evidence",
                ""
            )
        ).strip()

        generated_reply = str(
            row.get(
                "generated_reply",
                ""
            )
        ).strip()

        print("\n" + "-" * 60)

        print(
            f"[{i}/{len(remaining_df)}] "
            f"Sample ID: {sample_id}"
        )

        # ----------------------------------------------------
        # Retry loop
        # ----------------------------------------------------

        max_retries = 5

        for attempt in range(
            1,
            max_retries + 1
        ):

            try:

                result = judge_reply(
                    customer_message=
                        customer_message,
                    predicted_intent=
                        predicted_intent,
                    historical_evidence=
                        historical_evidence,
                    generated_reply=
                        generated_reply,
                )

                output_row = {
                    "sample_id":
                        sample_id,

                    "correctness":
                        result["correctness"],

                    "groundedness":
                        result["groundedness"],

                    "helpfulness":
                        result["helpfulness"],

                    "tone":
                        result["tone"],

                    "hallucination_free":
                        result["hallucination_free"],

                    "overall":
                        result["overall"],

                    "judge_reason":
                        result["judge_reason"],
                }

                results.append(
                    output_row
                )

                # Save after EVERY successful judgment.
                save_results(results)

                print(
                    "Scores:",
                    {
                        "correctness":
                            result["correctness"],
                        "groundedness":
                            result["groundedness"],
                        "helpfulness":
                            result["helpfulness"],
                        "tone":
                            result["tone"],
                        "hallucination_free":
                            result[
                                "hallucination_free"
                            ],
                        "overall":
                            result["overall"],
                    }
                )

                break

            except Exception as e:

                error_text = str(e)

                print(
                    f"Attempt {attempt}/{max_retries} failed:"
                )

                print(error_text)

                # ------------------------------------------------
                # Temporary provider / budget errors
                # ------------------------------------------------

                retryable = (
                    "402" in error_text
                    or "429" in error_text
                    or "in_flight_budget_exhausted"
                    in error_text
                    or "rate limit"
                    in error_text.lower()
                )

                if retryable:

                    wait_seconds = min(
                        120 * attempt,
                        300
                    )

                    print(
                        f"Waiting {wait_seconds}s "
                        "before retry..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                else:

                    # Non-retryable error.
                    print(
                        "Non-retryable error."
                    )

                    break

        else:

            print(
                f"Could not judge sample "
                f"{sample_id} after retries."
            )

        # Small spacing between requests.
        time.sleep(1)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    score_columns = [
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "hallucination_free",
        "overall",
    ]

    print("\n" + "=" * 60)
    print("                JUDGING COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal rows: {len(results_df)}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print("\nMean scores:")

    for column in score_columns:

        mean_value = pd.to_numeric(
            results_df[column],
            errors="coerce"
        ).mean()

        if pd.notna(mean_value):

            print(
                f"{column:20s}: "
                f"{mean_value:.2f}/5"
            )

    successful = 0

    for _, row in results_df.iterrows():

        if is_valid_judgment(row):

            successful += 1

    failed = (
        len(results_df)
        - successful
    )

    print(
        f"\nSuccessful judgments: "
        f"{successful}"
    )

    print(
        f"Failed judgments: "
        f"{failed}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()