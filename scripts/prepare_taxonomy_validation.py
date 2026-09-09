import json
import random
import csv
from pathlib import Path


INPUT_FILE = Path("data/spotify_conversations.json")
OUTPUT_FILE = Path("data/taxonomy_validation_sample.csv")

SAMPLE_SIZE = 800
RANDOM_SEED = 42


def extract_customer_messages(conversation):
    """
    Tries to support common conversation JSON structures.

    Expected message fields:
      - text
      - inbound

    Returns a list of customer messages from one conversation.
    """

    if isinstance(conversation, dict):
        # Most likely structure: {"messages": [...]}
        if "messages" in conversation:
            messages = conversation["messages"]

        # Alternative: {"conversation": [...]}
        elif "conversation" in conversation:
            messages = conversation["conversation"]

        else:
            return []

    elif isinstance(conversation, list):
        messages = conversation

    else:
        return []

    customer_messages = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        text = str(message.get("text", "")).strip()

        if not text:
            continue

        inbound = message.get("inbound")

        # Dataset convention:
        # inbound=True => customer
        if inbound is True:
            customer_messages.append({
                "text": text,
                "tweet_id": message.get("tweet_id"),
            })

    return customer_messages


def choose_one_message(conversation):
    """
    Select one representative customer message.

    Preference:
      1. Avoid extremely short acknowledgements when possible.
      2. Prefer messages with enough information to infer intent.
    """

    messages = extract_customer_messages(conversation)

    if not messages:
        return None

    useful = [
        m for m in messages
        if len(m["text"].split()) >= 4
    ]

    if useful:
        return random.choice(useful)

    return random.choice(messages)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("Loading conversations...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Support formats such as {"conversations": [...]}
        if "conversations" in data:
            conversations = data["conversations"]
        else:
            raise ValueError(
                "JSON is a dictionary but no 'conversations' key was found."
            )
    elif isinstance(data, list):
        conversations = data
    else:
        raise ValueError("Unexpected JSON structure.")

    print(f"Total conversations: {len(conversations):,}")

    random.seed(RANDOM_SEED)

    # Shuffle conversation indices rather than customer tweets.
    indices = list(range(len(conversations)))
    random.shuffle(indices)

    rows = []

    for conversation_idx in indices:

        conversation = conversations[conversation_idx]

        message = choose_one_message(conversation)

        if message is None:
            continue

        rows.append({
            "sample_id": len(rows) + 1,
            "conversation_index": conversation_idx,
            "tweet_id": message.get("tweet_id"),
            "text": message["text"],
            "label": "",
            "confidence": "",
            "notes": "",
        })

        if len(rows) >= SAMPLE_SIZE:
            break

    if len(rows) < SAMPLE_SIZE:
        print(
            f"WARNING: only found {len(rows)} usable messages."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "conversation_index",
                "tweet_id",
                "text",
                "label",
                "confidence",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Validation sample created: {OUTPUT_FILE}")
    print(f"Messages sampled: {len(rows):,}")


if __name__ == "__main__":
    main()