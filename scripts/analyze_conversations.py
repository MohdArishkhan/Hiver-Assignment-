import json
import re
from collections import Counter

FILE = "data/spotify_conversations.json"

print("Loading conversations...")

with open(FILE, "r", encoding="utf-8") as f:
    conversations = json.load(f)

print("Total conversations:", len(conversations))


# -----------------------------------------
# Extract customer messages
# -----------------------------------------

customer_messages = []

for conversation in conversations:

    for message in conversation["messages"]:

        if message["author_id"].lower() != "spotifycares":

            text = message["text"]

            if text:
                customer_messages.append(text)


print("Customer messages:", len(customer_messages))


# -----------------------------------------
# Basic statistics
# -----------------------------------------

lengths = [
    len(text.split())
    for text in customer_messages
]

print()
print("Average words:", round(sum(lengths) / len(lengths), 2))
print("Shortest message:", min(lengths))
print("Longest message:", max(lengths))


# -----------------------------------------
# Common words
# -----------------------------------------

words = []

for text in customer_messages:

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words.extend(text.split())


stopwords = {
    "the", "a", "an", "is", "i", "im",
    "to", "and", "of", "it", "my",
    "on", "for", "in", "this", "that",
    "you", "me", "be", "with", "but",
    "have", "has", "do", "can", "are",
    "not", "just", "please", "spotify"
}

words = [
    word
    for word in words
    if word not in stopwords
    and len(word) > 2
]

counter = Counter(words)


print()
print("Top 100 words:")
print()

for word, count in counter.most_common(100):
    print(f"{word:20} {count}")


# -----------------------------------------
# Save customer messages
# -----------------------------------------

with open(
    "data/spotify_customer_messages.txt",
    "w",
    encoding="utf-8"
) as f:

    for message in customer_messages:
        f.write(message.strip())
        f.write("\n")


print()
print("Saved: data/spotify_customer_messages.txt")




# -----------------------------------------
# Random sample of customer messages
# -----------------------------------------

print()
print("=" * 80)
print("RANDOM CUSTOMER MESSAGE SAMPLE")
print("=" * 80)

sample = customer_messages[:]

import random

random.seed(42)

random.shuffle(sample)

for i, message in enumerate(sample[:200], 1):

    print()
    print(f"{i}. {message}")