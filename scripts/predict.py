import sys
import joblib


MODEL_FILE = "models/final_model.pkl"


def predict_message(text):
    model = joblib.load(MODEL_FILE)

    prediction = model.predict([text])[0]

    print("\n======================================")
    print("       SPOTIFY INTENT CLASSIFIER")
    print("======================================")
    print("Message:", text)
    print("Predicted Intent:", prediction)
    print("======================================\n")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter Spotify support message: ")

    text = text.strip()

    if not text:
        print("Please enter a message.")
        sys.exit(1)

    predict_message(text)