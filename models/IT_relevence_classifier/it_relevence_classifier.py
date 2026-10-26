
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "relevance_tickets.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "relevance_classifier.pkl"

# 1. Load dataset
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# 2. Validate columns
required_columns = {"text", "relevance"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}. "
        f"Found: {df.columns.tolist()}"
    )

df = df[["text", "relevance"]].copy()

# 3. Clean data
df["text"] = df["text"].fillna("").astype(str).str.strip()
df["relevance"] = df["relevance"].fillna("").astype(str).str.strip().str.upper()

df = df[df["text"] != ""]
df = df[df["relevance"].isin(["YES", "NO"])]

# Remove exact duplicate text entries
df = df.drop_duplicates(subset=["text"], keep="first")

if df["relevance"].nunique() != 2:
    raise ValueError("Dataset must contain both YES and NO labels.")

print("Dataset size:", len(df))
print("\nClass distribution:")
print(df["relevance"].value_counts())

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["relevance"],
    test_size=0.20,
    random_state=42,
    stratify=df["relevance"],
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Build and train the model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix (labels: NO, YES):")
print(confusion_matrix(y_test, y_pred, labels=["NO", "YES"]))

# 7. Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("\nModel saved to:", MODEL_FILE)
