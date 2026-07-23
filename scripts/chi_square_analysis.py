import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
import numpy as np
import os


# Create output directory

output_dir = "results/statistical_mining"

os.makedirs(output_dir, exist_ok=True)


# Load Dataset

df = pd.read_csv("processed_youtube_data_v2.csv")


# Category Mapping

category_mapping = {
    1: "Film and Animation",
    10: "Music",
    15: "Pets and Animals",
    17: "Sports",
    20: "Gaming",
    23: "Comedy",
    24: "Entertainment",
    26: "News and Politics"
}

df["category_name"] = df["category_id"].map(category_mapping)


# Performance Class Mapping

performance_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

df["performance_label"] = (
    df["performance_class"]
    .map(performance_mapping)
)


# Contingency Table

contingency_table = pd.crosstab(
    df["category_name"],
    df["performance_label"]
)


print("\n==============================")
print("Contingency Table")
print("==============================")

print(contingency_table)



# Chi-Square Test

chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)


print("\n==============================")
print("Chi-Square Test Result")
print("==============================")

print(f"Chi-Square Statistic : {chi2:.4f}")
print(f"Degrees of Freedom   : {dof}")
print(f"P-value              : {p_value:.6f}")



# Hypothesis Decision

alpha = 0.05


print("\n==============================")
print("Hypothesis Testing")
print("==============================")


if p_value < alpha:

    print("Reject H0")
    print(
        "There is a significant relationship "
        "between category and performance class."
    )

else:

    print("Fail to reject H0")
    print(
        "No significant relationship was found."
    )



# Expected Frequencies

expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)


print("\n==============================")
print("Expected Frequencies")
print("==============================")

print(expected_df.round(2))



# Cramer's V

n = contingency_table.sum().sum()

minimum_dimension = min(
    contingency_table.shape
) - 1


cramers_v = np.sqrt(
    (chi2 / n) /
    minimum_dimension
)


print("\n==============================")
print("Effect Size")
print("==============================")


print(
    f"Cramer's V : {cramers_v:.4f}"
)


if cramers_v < 0.1:

    strength = "Weak"

elif cramers_v < 0.3:

    strength = "Moderate"

elif cramers_v < 0.5:

    strength = "Strong"

else:

    strength = "Very Strong"


print(
    f"Association Strength : {strength}"
)


# Heatmap
plt.figure(figsize=(10,7))


sns.heatmap(
    contingency_table,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title(
    "Category vs Performance Class"
)

plt.xlabel(
    "Performance Class"
)

plt.ylabel(
    "Video Category"
)


plt.tight_layout()


heatmap_path = os.path.join(
    output_dir,
    "category_performance_heatmap.png"
)


plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nSaved: {heatmap_path}"
)



# Stacked Bar Chart
stacked_data = (
    contingency_table
    .copy()
)


stacked_data.plot(
    kind="bar",
    stacked=True,
    figsize=(12,7)
)


plt.title(
    "Performance Class Distribution by Category"
)

plt.xlabel(
    "Video Category"
)

plt.ylabel(
    "Number of Videos"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.legend(
    title="Performance Class"
)


plt.tight_layout()


bar_path = os.path.join(
    output_dir,
    "category_performance_stacked_bar.png"
)


plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {bar_path}"
)



print("\nStatistical Mining Completed.")