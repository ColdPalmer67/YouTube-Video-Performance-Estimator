"""
YouTube Video Performance Estimator

Flask backend for serving the trained CNN model.
The model predicts video performance into three classes:
Low, Medium, and High.

Model inputs:
- Text features:
  Title, tags, and description converted into padded sequences.

- Metadata features:
  Subscriber count, video duration, and category ID.
"""

import pickle
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Configuration

MODEL_PATH = "models/youtube_cnn_v2_best.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
SCALER_PATH = "models/scaler.pkl"
CATEGORY_PATH = "models/category_mapping.json"

MAX_SEQUENCE_LENGTH = 200


# Model output order
CLASS_LABELS = [
    "Low",
    "Medium",
    "High"
]


# Flask Setup

app = Flask(__name__)


# Load Model and Artifacts

print("Loading model and preprocessing files...")


model = load_model(MODEL_PATH)


with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# Optional category mapping
try:
    with open(CATEGORY_PATH, "r") as f:
        category_mapping = json.load(f)

except FileNotFoundError:
    category_mapping = {}


print("Model and preprocessing files loaded successfully.")


# Preprocessing Function

def preprocess_input(
        title,
        tags,
        description,
        subscriber_count,
        duration_seconds,
        category_id
):

    # Text branch

    combined_text = (
        str(title)
        + " "
        + str(tags)
        + " "
        + str(description)
    )


    sequence = tokenizer.texts_to_sequences(
        [combined_text]
    )


    padded_text = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )


    # Metadata branch
    # Order MUST match training

    metadata = np.array(
        [[
            subscriber_count,
            duration_seconds,
            category_id
        ]],
        dtype=float
    )


    metadata_scaled = scaler.transform(metadata)


    return padded_text, metadata_scaled



# Routes


@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route("/predict-form")
def predict_form():

    return render_template(
        "predict.html"
    )



@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get user input

        title = request.form.get(
            "title",
            ""
        )

        tags = request.form.get(
            "tags",
            ""
        )

        description = request.form.get(
            "description",
            ""
        )


        subscriber_count = float(
            request.form.get(
                "subscriber_count"
            )
        )


        duration_seconds = float(
            request.form.get(
                "duration_seconds"
            )
        )


        category_id = float(
            request.form.get(
                "category_id"
            )
        )


        # Preprocess

        text_input, metadata_input = preprocess_input(
            title,
            tags,
            description,
            subscriber_count,
            duration_seconds,
            category_id
        )


        # Prediction

        prediction = model.predict(
            {
                "text_input": text_input,
                "metadata_input": metadata_input
            }
        )


        predicted_index = int(
            np.argmax(
                prediction[0]
            )
        )


        predicted_label = CLASS_LABELS[
            predicted_index
        ]


        confidence = float(
            np.max(
                prediction[0]
            )
        ) * 100



        return render_template(
            "result.html",
            label=predicted_label,
            confidence=round(confidence, 2),
            title=title
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )



@app.route("/charts")
def charts():

    return render_template(
        "charts.html"
    )



@app.route("/project-insights")
def project_insights():

    return render_template(
        "project_insights.html"
    )



# Run Flask

if __name__ == "__main__":
    app.run(debug=True)