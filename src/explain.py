import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import os

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("data/final_dataset.csv")

# Convert category_id to numeric
df["category_id"] = df["category_id"].astype("category").cat.codes

features = [
    "title_length",
    "word_count",
    "has_number",
    "has_question",
    "uppercase_ratio",
    "publish_hour",
    "category_id",
    "channel_avg_views",
    "channel_avg_likes",
    "channel_total_trending",
    "is_weekend"
]

X = df[features]

# ======================================================
# Load Model
# ======================================================

model = joblib.load("models/best_model.pkl")

# ======================================================
# Feature Importance
# ======================================================

importance = model.feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })
    .sort_values(by="Importance", ascending=False)
)

print(feature_importance)

# Save CSV
os.makedirs("outputs", exist_ok=True)
feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# ======================================================
# Plot Feature Importance
# ======================================================

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()

# ======================================================
# SHAP
# ======================================================

explainer = shap.TreeExplainer(model)

sample = X.sample(500, random_state=42)

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

plt.savefig("outputs/shap_summary.png")

print("\nExplainability completed successfully!")