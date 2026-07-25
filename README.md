# YouTube Video Performance Estimator

**Live Demo:**  
https://you-tube-video-performance-estimato.vercel.app/

A machine learning web application that predicts YouTube video performance classes (**Low, Medium, High**) using natural language processing (NLP) and video metadata features.

This project was developed as a **Final Year Project** to explore the combination of **data mining, machine learning, deep learning, and software engineering** through an end-to-end application.

> ⚠️ Note: This project is built for academic and portfolio demonstration purposes. The predictions are experimental estimates from a trained machine learning model and should not be interpreted as guaranteed indicators of future video success.

---

# Project Overview

The system analyzes YouTube video information and estimates its expected performance class based on:

- Video title
- Tags
- Description
- Subscriber count
- Video duration
- Category

The goal is to demonstrate how textual and numerical signals can be combined to build a machine learning-based prediction system.

---

# Live Application Features

The deployed web application allows users to:

1. Enter YouTube video information
2. Submit video details for prediction
3. Receive predicted performance class:
   - Low
   - Medium
   - High
4. View model confidence score
5. Explore:
   - Dataset insights
   - Statistical analysis
   - Model evaluation results
   - Similar video examples

---

# Machine Learning Features

## NLP Text Analysis

The model processes:

- Video title
- Tags
- Description

Text features are transformed using tokenization and padding before being passed into the deep learning model.

## Metadata Integration

The model also uses numerical video information:

- Subscriber count
- Video duration
- Category ID

These features provide additional context beyond text information.

---

# Machine Learning Pipeline

```text
YouTube Data Collection
          ↓
Data Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Statistical Mining (Chi-Square Analysis)
          ↓
Baseline Machine Learning Models
          ↓
CNN Deep Learning Model
          ↓
Flask Web Application
          ↓
Deployment
```

---

# Model Architecture

The final prediction model uses a **multi-input CNN architecture**.

## Text Branch

Processes combined text:

- Title
- Tags
- Description

Pipeline:

```
Text Input
    ↓
Tokenization
    ↓
Padding
    ↓
Embedding Layer
    ↓
1D CNN Feature Extraction
```

## Metadata Branch

Processes numerical features:

- Subscriber count
- Duration
- Category ID

The extracted text and metadata features are combined using **late fusion** before classification.

Final output:

- Low Performance
- Medium Performance
- High Performance

---

# Model Evaluation Dashboard

The application includes evaluation pages showing:

- Exploratory data analysis results
- Category-performance relationships
- Chi-square statistical analysis
- Baseline model comparison
- Confusion matrices
- CNN training performance curves

---

# Tech Stack

## Programming

- Python

## Machine Learning / Data Mining

- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

## Web Development

- Flask
- HTML
- CSS
- JavaScript

## Data Source

- YouTube Data API

## Deployment

- Vercel

---

# Project Structure

```text
YouTube-Video-Performance-Estimator/

├── app.py                  # Flask application
├── models/                 # Trained model and preprocessing artifacts
├── scripts/                # Data collection and analysis scripts
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript, and visual assets
├── data/                   # Supporting dataset files
└── requirements.txt        # Python dependencies
```

---

# Academic Context

This repository contains the implementation of my Final Year Project.

The project demonstrates an end-to-end workflow involving:

- Data collection
- Data preprocessing
- Exploratory analysis
- Statistical testing
- Machine learning experimentation
- Deep learning development
- Web application deployment

---

# Current Limitations

This project is a research and learning implementation rather than a production forecasting system.

Current limitations include:

- Prediction accuracy is limited by available dataset signals
- YouTube performance depends on many external factors not included in the model
- The model does not account for future trends, audience behaviour, or platform algorithm changes
- Predictions should be interpreted as model estimates rather than guarantees

---

# Future Improvements

Possible future improvements:

- Increase dataset size and diversity
- Improve prediction performance
- Experiment with transformer-based NLP models
- Add real-time YouTube API analysis
- Improve similarity search recommendations
- Deploy with a production-scale ML infrastructure

---

# Disclaimer

This project was created for **academic and portfolio demonstration purposes**.

The prediction results are generated by a trained machine learning model and should not be considered professional advice or a guaranteed prediction of YouTube video success.