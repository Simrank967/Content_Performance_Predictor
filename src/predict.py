import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================
# Load Model
# ==========================================

model = joblib.load("models/best_model.pkl")

# Historical dataset (for channel statistics)
df = pd.read_csv("data/final_dataset.csv")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# User Input
# ==========================================

title = "Who is SRK?"
channel = "BBC News"
publish_hour = 18
category_id = 24
is_weekend = 0

# ==========================================
# Generate Embedding
# ==========================================

embedding = embedding_model.encode([title])[0]

# ==========================================
# Channel Statistics
# ==========================================

if channel in df["channel_title"].values:

    channel_data = df[df["channel_title"] == channel]

    avg_views = channel_data["views"].mean()
    avg_likes = channel_data["likes"].mean()
    total_trending = len(channel_data)

else:

    avg_views = df["views"].mean()
    avg_likes = df["likes"].mean()
    total_trending = 1

# ==========================================
# Build Feature Vector
# ==========================================

sample = {}

for i in range(len(embedding)):
    sample[f"title_emb_{i}"] = embedding[i]

sample["publish_hour"] = publish_hour
sample["category_id"] = category_id
sample["channel_avg_views"] = avg_views
sample["channel_avg_likes"] = avg_likes
sample["channel_total_trending"] = total_trending
sample["is_weekend"] = is_weekend

sample = pd.DataFrame([sample])

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("=" * 50)

if prediction == 1:
    print("Prediction : HIGH ENGAGEMENT")
else:
    print("Prediction : LOW ENGAGEMENT")

print(f"Confidence : {probability:.2%}")