"""
YouTube Video Performance Estimator

Flask backend for serving the trained CNN model using ONNX Runtime.
The model predicts video performance into three classes:
Low, Medium, and High.

Model inputs:
- Text features:
  Title, tags, and description converted into padded sequences.

- Metadata features:
  Subscriber count, video duration, and category ID.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import onnxruntime as ort

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
ONNX_MODEL_PATH = "models/model.onnx"
TOKENIZER_PATH = "models/tokenizer.pkl"
SCALER_PATH = "models/scaler.pkl"
CATEGORY_PATH = "models/category_mapping.json"
DATA_PATH = "data/processed_youtube_data_v2.csv"

MAX_SEQUENCE_LENGTH = 200
CLASS_LABELS = ["Low", "Medium", "High"]

# Flask Setup
app = Flask(__name__)

print("Loading ONNX model and preprocessing files...")
# Initialize ONNX Inference Session
session = ort.InferenceSession(ONNX_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

try:
    with open(CATEGORY_PATH, "r") as f:
        category_mapping = json.load(f)
except FileNotFoundError:
    category_mapping = {}

print("Model and preprocessing files loaded successfully.")

# For Similar Video Search
print("Loading dataset...")
dataset = pd.read_csv(DATA_PATH)
dataset["combined_text"] = (
    dataset["title"].fillna("") + " " +
    dataset["tags"].fillna("") + " " +
    dataset["description"].fillna("")
)
print("Dataset loaded successfully.")

# TF-IDF Similarity Index
print("Building similarity index...")
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
dataset_vectors = tfidf.fit_transform(dataset["combined_text"])
print("Similarity index ready.")


def pad_sequences_custom(sequences, maxlen=200):
    """
    Pure NumPy replacement for keras.preprocessing.sequence.pad_sequences.
    Pads and truncates post-sequence to match training format.
    """
    arr = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            truncated = seq[:maxlen]
            arr[i, :len(truncated)] = truncated
    return arr


def preprocess_input(title, tags, description, subscriber_count, duration_seconds, category_id):
    combined_text = f"{title} {tags} {description}"
    sequence = tokenizer.texts_to_sequences([combined_text])
    
    # Custom post-padding and truncation
    padded_text = pad_sequences_custom(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    metadata = np.array([[subscriber_count, duration_seconds, category_id]], dtype=float)
    metadata_scaled = scaler.transform(metadata).astype(np.float32)

    return padded_text, metadata_scaled


def find_similar_videos(title, tags, description, subscriber_count, duration_seconds, category_id, top_n=3):
    input_text = f"{title} {tags} {description}"
    input_vector = tfidf.transform([input_text])
    text_scores = cosine_similarity(input_vector, dataset_vectors)[0]

    subscriber_diff = abs(dataset["subscriber_count"] - subscriber_count) / (dataset["subscriber_count"].max() + 1)
    duration_diff = abs(dataset["duration_seconds"] - duration_seconds) / (dataset["duration_seconds"].max() + 1)
    
    metadata_score = 1 - ((subscriber_diff + duration_diff) / 2)
    category_score = (dataset["category_id"] == int(category_id)).astype(float)

    final_score = (text_scores * 0.7) + (metadata_score * 0.2) + (category_score * 0.1)

    results = dataset.copy()
    results["similarity"] = final_score * 100
    results = results.sort_values(by="similarity", ascending=False)

    class_map = {0: "Low", 1: "Medium", 2: "High"}
    videos = []

    for _, row in results.head(top_n).iterrows():
        videos.append({
            "title": row["title"],
            "performance_class": class_map.get(int(row["performance_class"]), "Unknown"),
            "category_id": row["category_id"],
            "subscriber_count": int(row["subscriber_count"]),
            "duration_seconds": int(row["duration_seconds"]),
            "similarity": round(row["similarity"], 2),
            "youtube_url": f"https://www.youtube.com/watch?v={row['video_id']}"
        })

    return videos


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-form")
def predict_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        title = request.form.get("title", "")
        tags = request.form.get("tags", "")
        description = request.form.get("description", "")
        subscriber_count = float(request.form.get("subscriber_count", 0))
        duration_seconds = float(request.form.get("duration_seconds", 0))
        category_id = float(request.form.get("category_id", 0))

        text_input, metadata_input = preprocess_input(
            title, tags, description, subscriber_count, duration_seconds, category_id
        )

        # Map inputs to ONNX model input names
        input_feed = {
            session.get_inputs()[0].name: text_input,
            session.get_inputs()[1].name: metadata_input
        }

        # Run ONNX inference
        prediction = session.run(None, input_feed)[0]

        predicted_index = int(np.argmax(prediction[0]))
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = float(np.max(prediction[0])) * 100

        similar_videos = find_similar_videos(
            title, tags, description, subscriber_count, duration_seconds, category_id
        )

        return render_template(
            "result.html",
            label=predicted_label,
            confidence=round(confidence, 2),
            title=title,
            similar_videos=similar_videos
        )

    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {str(e)}")


@app.route("/charts")
def charts():
    return render_template("charts.html")


@app.route("/project-insights")
def project_insights():
    return render_template("project_insights.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)