import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)



RESULT_PATH = "results/machine_learning"

os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



print("Loading dataset...")

df = pd.read_csv(
    "processed_youtube_data_v2.csv"
)


print("\nDataset Shape:")
print(df.shape)



print("\nPreparing text features...")


df["combined_text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["tags"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
)


X_text = df["combined_text"]

X_numeric = df[
    [
        "subscriber_count",
        "duration_seconds",
        "category_id"
    ]
]


y = df["performance_class"]


# Train-Test Split

(
    X_text_train,
    X_text_test,
    X_num_train,
    X_num_test,
    y_train,
    y_test
) = train_test_split(
    X_text,
    X_numeric,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain size:", len(y_train))
print("Test size :", len(y_test))



# TF-IDF

print("\nApplying TF-IDF...")


tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)


X_text_train_tfidf = tfidf.fit_transform(
    X_text_train
)


X_text_test_tfidf = tfidf.transform(
    X_text_test
)



# Numeric Scaling

print("Scaling numerical features...")


scaler = StandardScaler()


X_num_train_scaled = scaler.fit_transform(
    X_num_train
)


X_num_test_scaled = scaler.transform(
    X_num_test
)


# Combine Features

X_train = hstack(
    [
        X_text_train_tfidf,
        X_num_train_scaled
    ]
)


X_test = hstack(
    [
        X_text_test_tfidf,
        X_num_test_scaled
    ]
)


# Models


models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),


    "Random Forest":
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

}



# Cross Validation

print("\n==============================")
print("5-Fold Cross Validation")
print("==============================")


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )


    print("\n", name)

    print(
        "Fold Accuracy:",
        scores
    )

    print(
        "Mean Accuracy:",
        scores.mean()
    )

    print(
        "Std:",
        scores.std()
    )


# Final Training + Evaluation


results = []


for name, model in models.items():


    print("\n==============================")
    print(name)
    print("==============================")


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Low",
                "Medium",
                "High"
            ]
        )
    )


    print("Accuracy:", accuracy)


    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1
        ]
    )



    # Confusion Matrix


    cm = confusion_matrix(
        y_test,
        predictions
    )


    plt.figure(
        figsize=(6,5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Low",
            "Medium",
            "High"
        ],
        yticklabels=[
            "Low",
            "Medium",
            "High"
        ]
    )


    plt.title(
        f"{name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.tight_layout()


    filename = (
        name
        .lower()
        .replace(" ","_")
        +
        "_confusion_matrix.png"
    )


    plt.savefig(
        os.path.join(
            RESULT_PATH,
            filename
        ),
        dpi=300
    )


    plt.close()




results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
)


print("\n==============================")
print("Model Comparison")
print("==============================")


print(results_df)



results_df.to_csv(
    os.path.join(
        RESULT_PATH,
        "model_comparison.csv"
    ),
    index=False
)


print(
    "\nMachine Learning Baseline Completed."
)