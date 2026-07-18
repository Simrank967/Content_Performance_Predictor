import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from catboost import CatBoostClassifier

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("data/final_dataset.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")




# Save category mapping
category_dtype = df["category_id"].astype("category")

category_mapping = list(category_dtype.cat.categories)

joblib.dump(
    category_mapping,
    "models/category_id_categories.pkl"
)

# Encode using the saved mapping
df["category_id"] = category_dtype.cat.codes

# ======================================================
# Feature Selection
# ======================================================

# Title embedding features
embedding_features = [
    col for col in df.columns
    if col.startswith("title_emb_")
]

# Metadata features
metadata_features = [
    "publish_hour",
    "category_id",
    "channel_avg_views",
    "channel_avg_likes",
    "channel_total_trending",
    "is_weekend"
]

features = embedding_features + metadata_features
target = "high_engagement"

X = df[features]
y = df[target]

print("\nFeature Summary")
print("-" * 40)
print(f"Embedding Features : {len(embedding_features)}")
print(f"Metadata Features  : {len(metadata_features)}")
print(f"Total Features     : {len(features)}")

print("\nFirst 5 Embedding Columns:")
print(embedding_features[:5])

# ======================================================
# Data Cleaning
# ======================================================

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("\nNaN values remaining:", X.isnull().sum().sum())

# ======================================================
# Train-Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Samples :", len(X_train))
print("Test Samples  :", len(X_test))

# ======================================================
# Models
# ======================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )

}

results = {}

best_model = None
best_model_name = None
best_score = 0

# ======================================================
# Training
# ======================================================

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc
    }

    if roc > best_score:
        best_score = roc
        best_model = model
        best_model_name = name

# ======================================================
# Results Summary
# ======================================================

results_df = pd.DataFrame(results).T

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# ======================================================
# Save Outputs
# ======================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
results_df.to_csv("outputs/model_comparison.csv")

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best Model : {best_model_name}")
print(f"Best ROC-AUC : {best_score:.4f}")
print("Saved Model : models/best_model.pkl")
print("Saved Results : outputs/model_comparison.csv")