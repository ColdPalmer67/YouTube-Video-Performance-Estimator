import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


FILE = "data/processed_youtube_data_v2.csv"
OUTPUT_DIR = "results/eda"


# Load category names from JSON file
with open("data/category_mapping.json", "r") as f:
    category_mapping = json.load(f)


def get_category_name(category_id):
    return category_mapping.get(str(category_id), "Unknown")


# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():

    df = pd.read_csv(
        FILE,
        keep_default_na=False
    )

    print("Dataset shape:")
    print(df.shape)

    return df



def basic_statistics(df):

    print("\nDataset information:")
    print(df.info())

    print("\nNumerical statistics:")

    print(
        df[
            [
                "view_count",
                "subscriber_count",
                "duration_seconds",
                "performance_score"
            ]
        ].describe()
    )



def category_analysis(df):

    print("\nCategory distribution:")

    category_count = (
        df["category_id"]
        .value_counts()
        .sort_index()
    )


    category_count.index = [
        get_category_name(x)
        for x in category_count.index
    ]


    print(category_count)


    plt.figure(figsize=(8,5))

    category_count.plot(
        kind="bar"
    )

    plt.title(
        "Number of Videos per Category"
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Number of Videos"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/category_distribution.png"
    )

    plt.close()



def class_analysis(df):

    class_count = (
        df["performance_class"]
        .value_counts()
        .sort_index()
    )


    print("\nPerformance class distribution:")
    print(class_count)


    plt.figure(figsize=(6,4))

    class_count.plot(
        kind="bar"
    )


    plt.title(
        "Performance Class Distribution"
    )

    plt.xlabel(
        "Performance Class"
    )

    plt.ylabel(
        "Number of Videos"
    )

    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_DIR}/performance_class_distribution.png"
    )

    plt.close()



def performance_distribution(df):

    plt.figure(figsize=(8,5))


    plt.hist(
        df["performance_score"],
        bins=50
    )


    plt.title(
        "Performance Score Distribution"
    )

    plt.xlabel(
        "Performance Score"
    )

    plt.ylabel(
        "Frequency"
    )


    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_DIR}/performance_score_distribution.png"
    )

    plt.close()



def category_performance(df):

    result = (
        df.groupby("category_id")["performance_score"]
        .mean()
        .sort_values(ascending=False)
    )


    result.index = [
        get_category_name(x)
        for x in result.index
    ]


    print("\nAverage performance score by category:")
    print(result)



def performance_boxplot(df):

    plt.figure(figsize=(8,5))


    sns.boxplot(
        data=df,
        x="category_id",
        y="performance_score"
    )


    plt.title(
        "Performance Score Distribution by Category"
    )

    plt.xlabel(
        "Category ID"
    )

    plt.ylabel(
        "Performance Score"
    )


    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_DIR}/performance_score_category_boxplot.png"
    )

    plt.close()



def correlation_analysis(df):

    numeric_data = df[
        [
            "view_count",
            "subscriber_count",
            "duration_seconds",
            "performance_score",
            "performance_class"
        ]
    ]


    correlation = numeric_data.corr()


    print("\nCorrelation matrix:")
    print(correlation)


    plt.figure(figsize=(8,6))


    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )


    plt.title(
        "Feature Correlation Heatmap"
    )


    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_DIR}/feature_correlation_heatmap.png"
    )

    plt.close()



if __name__ == "__main__":

    df = load_data()

    basic_statistics(df)

    category_analysis(df)

    class_analysis(df)

    performance_distribution(df)

    category_performance(df)

    performance_boxplot(df)

    correlation_analysis(df)

    print("\nEDA completed.")