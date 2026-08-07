import os
import csv
import json
import pickle
import numpy as np
import onnxruntime as ort

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.onnx")
WORD_INDEX_PATH = os.path.join(BASE_DIR, "models", "word_index.json")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
VECTORS_PATH = os.path.join(BASE_DIR, "models", "dataset_vectors.pkl")

CATEGORY_PATH = os.path.join(BASE_DIR, "data", "category_mapping.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_youtube_data_v2.csv")

MAX_SEQUENCE_LENGTH = 200
CLASS_LABELS = ["Low", "Medium", "High"]

app = Flask(__name__)

print("Loading ONNX model and preprocessing assets...")
session = ort.InferenceSession(ONNX_MODEL_PATH)

with open(WORD_INDEX_PATH, "r", encoding="utf-8") as f:
    word_index = json.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Loading pre-computed similarity index...")
with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(VECTORS_PATH, "rb") as f:
    dataset_vectors = pickle.load(f)

try:
    with open(CATEGORY_PATH, "r") as f:
        category_mapping = json.load(f)
except FileNotFoundError:
    category_mapping = {}

print("Loading dataset...")
dataset_rows = []

with open(DATA_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset_rows.append(row)

max_subscribers = max([float(r.get("subscriber_count", 1)) for r in dataset_rows] or [1]) + 1
max_duration = max([float(r.get("duration_seconds", 1)) for r in dataset_rows] or [1]) + 1

print("Assets loaded successfully.")


MODEL_VOCAB_SIZE = 10000
OOV_TOKEN_ID = 1


def preprocess_text(text, maxlen=200):
    words = str(text).lower().split()
    sequence = []

    for word in words:
        token_id = word_index.get(word, OOV_TOKEN_ID)

        if token_id >= MODEL_VOCAB_SIZE:
            token_id = OOV_TOKEN_ID

        sequence.append(token_id)

    padded = np.zeros(
        (1, maxlen),
        dtype=np.float32
    )

    if sequence:
        truncated = sequence[:maxlen]
        padded[0, :len(truncated)] = truncated

    return padded


def preprocess_input(title, tags, description, subscriber_count, duration_seconds, category_id):
    combined_text = f"{title} {tags} {description}"
    
    padded_text = preprocess_text(combined_text, maxlen=MAX_SEQUENCE_LENGTH)
    
    metadata = np.array([[subscriber_count, duration_seconds, category_id]], dtype=np.float32)
    metadata_scaled = scaler.transform(metadata).astype(np.float32)

    return padded_text, metadata_scaled


def find_similar_videos(title, tags, description, subscriber_count, duration_seconds, category_id, top_n=3):
    input_text = f"{title} {tags} {description}"
    input_vector = tfidf.transform([input_text])
    text_scores = cosine_similarity(input_vector, dataset_vectors)[0]

    class_map = {"0": "Low", "1": "Medium", "2": "High", 0: "Low", 1: "Medium", 2: "High"}
    scored_videos = []

    for idx, row in enumerate(dataset_rows):
        row_subs = float(row.get("subscriber_count", 0))
        row_dur = float(row.get("duration_seconds", 0))
        row_cat = float(row.get("category_id", 0))

        subscriber_diff = abs(row_subs - subscriber_count) / max_subscribers
        duration_diff = abs(row_dur - duration_seconds) / max_duration
        
        metadata_score = 1.0 - ((subscriber_diff + duration_diff) / 2.0)
        category_score = 1.0 if row_cat == float(category_id) else 0.0

        final_score = (text_scores[idx] * 0.7) + (metadata_score * 0.2) + (category_score * 0.1)
        similarity_pct = round(float(final_score) * 100, 2)

        raw_perf = row.get("performance_class", "0")
        perf_label = class_map.get(raw_perf, "Unknown")

        scored_videos.append({
            "title": row.get("title", "Untitled"),
            "performance_class": perf_label,
            "category_id": row.get("category_id", 0),
            "subscriber_count": int(row_subs),
            "duration_seconds": int(row_dur),
            "similarity": similarity_pct,
            "youtube_url": f"https://www.youtube.com/watch?v={row.get('video_id', '')}",
            "score": final_score
        })

    scored_videos.sort(key=lambda x: x["score"], reverse=True)
    return scored_videos[:top_n]


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

        input_names = [i.name for i in session.get_inputs()]
        input_feed = {
            input_names[0]: text_input,
            input_names[1]: metadata_input
        }

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
    app.run(host="0.0.0.0", port=port, debug=False)