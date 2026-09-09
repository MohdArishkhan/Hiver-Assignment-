import joblib

model = joblib.load("models/final_model.pkl")

tests = [
    "Spotify keeps buffering every few seconds",
    "I can't login to my Spotify account",
    "I was charged twice for Premium",
    "How can I download music for offline listening?",
]

for text in tests:
    prediction = model.predict([text])[0]
    print(f"{text} => {prediction}")