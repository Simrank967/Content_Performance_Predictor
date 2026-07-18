import pandas as pd
import os

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/USvideos.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

# -----------------------------
# Basic Information
# -----------------------------
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst Five Rows:")
print(df.head())

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates Removed.")

# -----------------------------
# Convert Publish Time
# -----------------------------
df["publish_time"] = pd.to_datetime(df["publish_time"])

# Create New Features
df["publish_date"] = df["publish_time"].dt.date
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_day"] = df["publish_time"].dt.day_name()
df["publish_month"] = df["publish_time"].dt.month_name()

# -----------------------------
# Clean Missing Values
# -----------------------------
df = df.dropna()

print("\nFinal Shape:")
print(df.shape)

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
os.makedirs("data", exist_ok=True)

OUTPUT_PATH = "data/cleaned_youtube.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print(f"Location: {OUTPUT_PATH}")
