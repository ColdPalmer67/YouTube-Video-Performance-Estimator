# YouTube Video Performance Estimator

A machine learning web application that predicts YouTube video performance classes (**Low, Medium, High**) using natural language processing (NLP) and video metadata features.

This project was developed as a **Final Year Project** to explore the application of data mining, machine learning, deep learning, and software engineering concepts in an end-to-end system.

---

## Project Overview

The system analyzes YouTube video information, including:

- Video title
- Tags
- Description
- Subscriber count
- Video duration
- Category

Using these features, the trained model estimates the expected performance class of a video.

---

## Features

- NLP-based text analysis using:
  - Video title
  - Tags
  - Description

- Multi-input CNN classification model combining:
  - Text features
  - Numerical metadata features

- Metadata integration:
  - Subscriber count
  - Video duration
  - Category ID

- Flask web application deployment

- Model evaluation dashboard containing:
  - Exploratory data analysis visualizations
  - Statistical analysis results
  - Machine learning comparison charts
  - Deep learning evaluation metrics

---

## Tech Stack

**Programming Language**
- Python

**Machine Learning & Data Mining**
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

**Web Development**
- Flask
- HTML
- CSS
- JavaScript

**Data Source**
- YouTube Data API

---

## Machine Learning Pipeline

```text
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Statistical Testing (Chi-Square Analysis)
        ↓
Baseline Machine Learning Models
        ↓
CNN Deep Learning Model
        ↓
Flask Web Deployment
```

---

## Model Architecture

The final model uses a multi-input CNN architecture.

### Text Branch
- Combines video title, tags, and description
- Tokenization and padding
- Embedding layer
- 1D Convolutional Neural Network

### Metadata Branch
Uses numerical features:

- Subscriber count
- Video duration
- Category ID

The extracted features from both branches are combined using late fusion before classification into:

- Low Performance
- Medium Performance
- High Performance

---

## Project Structure

```text
YouTube-Video-Performance-Estimator/

├── app.py                 # Flask application
├── models/                # Trained model and preprocessing artifacts
├── scripts/               # Data collection, preprocessing, and analysis scripts
├── templates/             # HTML templates
├── static/                # CSS, JavaScript, and visualization images
├── data/                  # Supporting project files
└── requirements.txt       # Python dependencies
```

---

## Application Features

The deployed Flask application allows users to:

1. Enter YouTube video information
2. Submit the video details for prediction
3. Receive a predicted performance category
4. View prediction confidence
5. Explore model evaluation and analysis charts

---

## Academic Context

This repository contains the implementation of my Final Year Project.

The project demonstrates an end-to-end machine learning workflow, including:

- Data collection
- Data preprocessing
- Statistical analysis
- Machine learning experimentation
- Deep learning model development
- Web application deployment

---

## Future Improvements

Possible future improvements include:

- Increasing dataset size
- Improving prediction performance
- Experimenting with transformer-based NLP models
- Adding live YouTube API prediction capability
- Deploying the application to a cloud platform

---

## Disclaimer

This project was created for academic and portfolio demonstration purposes. Prediction results are estimates produced by a trained machine learning model and should not be considered as guaranteed indicators of future video performance.