import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/cleaned_youtube.csv")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# ======================================
# Load Sentence Transformer
# ======================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    df["title"].fillna("").tolist(),
    show_progress_bar=True
)

embedding_df = pd.DataFrame(
    embeddings,
    columns=[f"title_emb_{i}" for i in range(384)]
)

df = pd.concat(
    [df.reset_index(drop=True), embedding_df],
    axis=1
)
print("Embeddings generated successfully!")
print(f"Embedding Shape: {embedding_df.shape}")

# =====================================================
# Handle Missing Values
# =====================================================

df["title"] = df["title"].fillna("")
df["channel_title"] = df["channel_title"].fillna("Unknown")

# =====================================================
# Ensure publish_day exists
# =====================================================

if "publish_day" not in df.columns:
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    df["publish_day"] = df["publish_time"].dt.day_name()

# =====================================================
# Safe Ratio Calculations
# =====================================================

views = df["views"].replace(0, np.nan)

df["engagement_rate"] = (
    (df["likes"] + df["comment_count"]) / views
).fillna(0)

df["like_view_ratio"] = (
    df["likes"] / views
).fillna(0)

df["comment_view_ratio"] = (
    df["comment_count"] / views
).fillna(0)



# =====================================================
# Channel Features
# =====================================================

df["channel_avg_views"] = (
    df.groupby("channel_title")["views"]
    .transform("mean")
)

df["channel_avg_likes"] = (
    df.groupby("channel_title")["likes"]
    .transform("mean")
)

df["channel_total_trending"] = (
    df.groupby("channel_title")["video_id"]
    .transform("count")
)

# =====================================================
# Weekend Feature
# =====================================================

df["is_weekend"] = (
    df["publish_day"]
    .isin(["Saturday", "Sunday"])
    .astype(int)
)

# =====================================================
# Target Variable
# =====================================================

threshold = df["engagement_rate"].quantile(0.80)

df["high_engagement"] = (
    df["engagement_rate"] >= threshold
).astype(int)

# =====================================================
# Replace Remaining NaNs
# =====================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns

df[numeric_cols] = df[numeric_cols].fillna(0)

# =====================================================
# Save Dataset
# =====================================================

df.to_csv("data/final_dataset.csv", index=False)

print("\nFeature Engineering Completed Successfully!")
print(f"Final Dataset Shape: {df.shape}")

print("\nNew Features Created:")

features = [
    "engagement_rate",
    "like_view_ratio",
    "comment_view_ratio",
    "channel_avg_views",
    "channel_avg_likes",
    "channel_total_trending",
    "is_weekend",
    "high_engagement"
]

for feature in features:
    print(f"✓ {feature}")

print("\nFirst Five Rows:")
print(df[features].head())