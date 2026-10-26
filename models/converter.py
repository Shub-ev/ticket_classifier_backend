"""Restore model artifacts encoded as base64 text files.

Run this utility from the directory containing the encoded model files.
"""

import base64

files = [
    "tfidf_vectorizer.txt",
    "tfidf_classifier.txt",
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        encoded = f.read()

    pkl_file = file.replace(".txt", ".pkl")
    with open(pkl_file, "wb") as f:
        f.write(base64.b64decode(encoded))

    print(f"Restored {pkl_file}")