import json
import pandas as pd
import html


INPUT_FILE = "data/spotify_conversations.json"
OUTPUT_FILE = "data/spotify_knowledge_base.csv"


# ==========================================
# LOAD CONVERSATIONS
# ==========================================

print("Loading Spotify conversations...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    conversations = json.load(f)

print("Total conversations:", len(conversations))


# ==========================================
# BUILD CUSTOMER -> AGENT PAIRS
# ==========================================

rows = []

for conversation in conversations:

    messages = conversation.get("messages", [])

    for i in range(len(messages) - 1):

        customer = messages[i]
        agent = messages[i + 1]

        # We want:
        # customer message -> SpotifyCares response

        if (
            customer.get("inbound") is True
            and agent.get("inbound") is False
        ):

            customer_text = customer.get("text", "").strip()
            agent_text = agent.get("text", "").strip()

            if not customer_text.strip():
                continue

            if not agent_text.strip():
                continue

            # Ignore rows where the customer message is basically only a mention
            clean_customer = customer_text.replace("@SpotifyCares", "").strip()

            if len(clean_customer) < 5:
                continue

            # Decode HTML entities such as &amp;
            customer_text = html.unescape(customer_text)
            agent_text = html.unescape(agent_text)

            rows.append({
                "customer_message": customer_text,
                "spotify_reply": agent_text,
                "customer_tweet_id": customer.get("tweet_id"),
                "spotify_reply_id": agent.get("tweet_id"),
            })


# ==========================================
# SAVE
# ==========================================

df = pd.DataFrame(rows)

df = df.drop_duplicates(
    subset=["customer_message", "spotify_reply"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("\n==========================================")
print("       KNOWLEDGE BASE CREATED")
print("==========================================")

print("Customer-reply pairs:", len(df))
print("Output:", OUTPUT_FILE)

print("\nSample pairs:\n")

for _, row in df.head(5).iterrows():

    print("------------------------------------------")
    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nSPOTIFY:")
    print(row["spotify_reply"])


print("\n==========================================")
print("DONE")
print("==========================================")