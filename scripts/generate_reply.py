import os
from openai import OpenAI

MODEL = "google/gemini-2.5-flash"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Set it in PowerShell using: $env:OPENROUTER_API_KEY='your-key-here'"
    )

# OpenRouter recommends passing HTTP-Referer and X-Title headers
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://github.com/MohdArishkhan/Hiver-Assignment-",
        "X-Title": "Spotify Support Agent",
    },
)

def generate_reply(customer_message: str, intent: str, evidence: list) -> str:
    """
    Drafts a grounded customer support reply based on retrieved historical interactions.
    """
    if not evidence:
        evidence_text = "No direct historical matches found.\n"
    else:
        evidence_blocks = []
        for i, case in enumerate(evidence, start=1):
            sim = case.get("similarity", 0.0)
            sim_display = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
            
            block = (
                f"CASE {i}\n"
                f"Customer message:\n{case.get('customer_message', '').strip()}\n\n"
                f"Historical Spotify reply:\n{case.get('spotify_reply', '').strip()}\n\n"
                f"Similarity:\n{sim_display}"
            )
            evidence_blocks.append(block)
            
        evidence_text = "\n\n---\n\n".join(evidence_blocks)

    prompt = f"""You are a customer-support reply assistant for Spotify.

Your task is to draft a concise and helpful response to the customer.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL SPOTIFY SUPPORT EVIDENCE:
{evidence_text}

RULES:
1. Use the historical Spotify responses as your primary evidence.
2. Do not invent Spotify policies, refunds, account actions, or guarantees.
3. Do not claim that you performed an account action.
4. Do not copy Twitter usernames or internal agent initials (e.g., /AY, /MU).
5. Do not include old or irrelevant URLs from the historical tweets.
6. Do not mention that you are an AI.
7. Do not mention the historical examples to the customer.
8. Keep the reply concise and professional.
9. Address the customer's actual issue.
10. Prefer troubleshooting steps or questions supported by the historical evidence.
11. If the historical evidence is insufficient, ask for more information instead of guessing.
12. Return ONLY the customer-facing reply.

Write a natural support response, not a copy of an old tweet."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write safe, grounded Spotify customer-support replies. "
                    "Never invent unsupported policies, actions, or guarantees."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    reply = response.choices[0].message.content
    if not reply or not reply.strip():
        raise RuntimeError("OpenRouter returned an empty response.")

    return reply.strip()