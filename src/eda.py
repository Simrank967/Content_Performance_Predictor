import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/cleaned_youtube.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Shape
print("\nDataset Shape:")
print(df.shape)

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(10,6))

plt.hist(df["views"], bins=50)

plt.title("Views Distribution")

plt.xlabel("Views")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/views_distribution.png")

plt.show()

plt.figure(figsize=(10,6))

plt.hist(df["likes"], bins=50)

plt.title("Likes Distribution")

plt.xlabel("Likes")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/likes_distribution.png")

plt.show()

plt.figure(figsize=(10,6))

plt.hist(df["comment_count"], bins=50)

plt.title("Comments Distribution")

plt.xlabel("Comments")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/comments_distribution.png")

plt.show()

plt.figure(figsize=(10,6))

hour_counts = df["publish_hour"].value_counts().sort_index()

plt.bar(hour_counts.index, hour_counts.values)

plt.title("Videos Published by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Videos")

plt.xticks(range(24))

plt.tight_layout()
plt.savefig("outputs/upload_hour_distribution.png")
plt.show()

day_order = [
    "Monday","Tuesday","Wednesday",
    "Thursday","Friday","Saturday","Sunday"
]

day_counts = (
    df["publish_day"]
    .value_counts()
    .reindex(day_order)
)

plt.figure(figsize=(10,6))

plt.bar(day_counts.index, day_counts.values)

plt.title("Videos Published by Day")
plt.xlabel("Day")
plt.ylabel("Number of Videos")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("outputs/upload_day_distribution.png")
plt.show()

category_counts = (
    df["category_id"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

plt.bar(category_counts.index.astype(str),
        category_counts.values)

plt.title("Top Categories")
plt.xlabel("Category ID")
plt.ylabel("Number of Videos")

plt.tight_layout()
plt.savefig("outputs/category_distribution.png")
plt.show()

top_channels = (
    df["channel_title"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12,6))

plt.barh(top_channels.index,
         top_channels.values)

plt.title("Top 10 Channels")
plt.xlabel("Trending Videos")

plt.tight_layout()
plt.savefig("outputs/top_channels.png")
plt.show()

df["engagement_rate"] = (
    df["likes"] + df["comment_count"]
) / df["views"]

plt.figure(figsize=(10,6))

plt.hist(df["engagement_rate"], bins=50)

plt.title("Engagement Rate Distribution")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("outputs/engagement_distribution.png")

plt.show()

plt.figure(figsize=(8,6))

corr = df[
    ["views","likes","comment_count","engagement_rate"]
].corr()

plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

sample = df.sample(5000, random_state=42)

plt.figure(figsize=(10,6))

plt.scatter(
    sample["views"],
    sample["likes"],
    alpha=0.4
)

plt.title("Views vs Likes")

plt.xlabel("Views")
plt.ylabel("Likes")

plt.tight_layout()

plt.savefig("outputs/views_vs_likes.png")

plt.show()