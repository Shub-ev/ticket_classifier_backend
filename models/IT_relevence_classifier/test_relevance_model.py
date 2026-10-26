
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_FILE = BASE_DIR / "models" / "relevance_classifier.pkl"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}. Train the model first."
    )

model = joblib.load(MODEL_FILE)

print("IT Ticket Relevance Classifier")
print("Enter a request to test it. Type 'exit' to quit.\n")

while True:
    text = input("Enter request: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        print("Please enter some text.\n")
        continue

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    confidence = class_probabilities[prediction] * 100

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

    if prediction == "YES":
        print("Action: Pass to assignment-group classifier")
    else:
        print("Action: Reject as out-of-scope or request clarification")

    print("-" * 50)
