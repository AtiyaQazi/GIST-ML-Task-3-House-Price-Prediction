import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# HOUSE PRICE PREDICTION - DATA ANALYSIS
print("=" * 60)
print("HOUSE PRICE PREDICTION - DATA ANALYSIS")
print("=" * 60)
# Create output directory if it doesn't exist
os.makedirs("screenshots", exist_ok=True)

# LOAD DATASET
df = pd.read_csv("data/housing.csv")
print("\nOriginal Dataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# DATA CLEANING
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()
# Remove rows with missing values
df = df.dropna()
print("\nDataset Shape After Cleaning:")
print(df.shape)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# BASIC INFORMATION
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# EDA VISUALIZATIONS

# 1. PRICE DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.histplot(
    df["Price"],
    kde=True
)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

# 2. INCOME VS PRICE
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Avg. Area Income",
    y="Price"
)
plt.title("Average Area Income vs House Price")
plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.tight_layout()
plt.savefig("income_vs_price.png")
plt.close()

# 3. HOUSE AGE VS PRICE
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Avg. Area House Age",
    y="Price"
)
plt.title("House Age vs House Price")
plt.xlabel("Average Area House Age")
plt.ylabel("House Price")
plt.tight_layout()
plt.savefig("age_vs_price.png")
plt.close()

# 4. ROOMS VS PRICE
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Avg. Area Number of Rooms",
    y="Price"
)
plt.title("Number of Rooms vs House Price")
plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.tight_layout()
plt.savefig("rooms_vs_price.png")
plt.close()

# CORRELATION
numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("\nEDA completed successfully!")
print("\nGenerated Visualizations:")
print("- price_distribution.png")
print("- income_vs_price.png")
print("- age_vs_price.png")
print("- rooms_vs_price.png")
print("- correlation_heatmap.png")