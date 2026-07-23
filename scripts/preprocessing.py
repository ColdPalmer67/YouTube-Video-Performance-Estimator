import os
import pandas as pd
import numpy as np
import isodate


INPUT_FILE = "raw_youtube_data_v2.csv"
CLEAN_FILE = "clean_youtube_data_v2.csv"
OUTPUT_FILE = "processed_youtube_data_v2.csv"


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Dataset file not found.")

    print("Loading dataset...")
    
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="cp1252")

    print("Initial rows:", len(df))

    return df


def clean_data(df):

    print("\nCleaning data...")

    # Remove duplicate videos
    df = df.drop_duplicates(subset=["video_id"])

    # Fill missing text values
    df["title"] = df["title"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["description"] = df["description"].fillna("")

    # Convert numeric columns
    numeric_cols = [
        "view_count",
        "subscriber_count"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid numeric rows
    df = df.dropna(
        subset=[
            "view_count",
            "subscriber_count",
            "duration_seconds"
        ]
    )

    df = df[
        (df["view_count"] > 0) &
        (df["subscriber_count"] > 0)
    ]

    print("Rows after cleaning:", len(df))

    return df


def convert_duration(df):

    print("\nConverting duration...")

    def duration_to_seconds(value):

        if pd.isna(value) or value == "":
            return 0

        try:
            return int(
                isodate.parse_duration(value).total_seconds()
            )

        except:
            return 0


    df["duration_seconds"] = df["duration_seconds"].apply(
        duration_to_seconds
    )


    df = df[df["duration_seconds"] > 0]

    print("Rows after duration check:", len(df))

    return df



def balance_categories(df):

    print("\nBalancing categories...")

    category_size = 3000

    balanced = []

    for category in df["category_id"].unique():

        category_data = df[
            df["category_id"] == category
        ]

        if len(category_data) >= category_size:

            sample = category_data.sample(
                n=category_size,
                random_state=42
            )

            balanced.append(sample)

        else:
            print(
                "Category",
                category,
                "has only",
                len(category_data),
                "records"
            )

    df = pd.concat(balanced)

    print("\nBalanced distribution:")
    print(df["category_id"].value_counts())

    return df



def create_target(df):

    print("\nCreating performance classes...")

    views = np.log1p(
        df["view_count"]
    )

    subs = np.log1p(
        df["subscriber_count"]
    )

    df["performance_score"] = views / subs


    df["performance_class"] = pd.qcut(
        df["performance_score"],
        q=3,
        labels=[0, 1, 2]
    )


    df["performance_class"] = (
        df["performance_class"]
        .astype(int)
    )
    
    df["title"] = df["title"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["description"] = df["description"].fillna("")

    print("\nClass distribution:")
    print(
        df["performance_class"]
        .value_counts()
        .sort_index()
    )


    return df



if __name__ == "__main__":


    df = load_data(INPUT_FILE)

    df = clean_data(df)

    df = convert_duration(df)


    df.to_csv(
        CLEAN_FILE,
        index=False
    )

    print("\nClean dataset saved.")


    df = balance_categories(df)

    df = create_target(df)


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\nProcessing complete.")
    print(
        "Final dataset:",
        df.shape
    )