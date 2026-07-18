import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
from sentence_transformers import SentenceTransformer
# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="Content Performance Predictor",
    layout="wide"
)

# =====================================
# Load Model
# =====================================

model = joblib.load("models/best_model.pkl")
category_mapping = joblib.load("models/category_id_categories.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/final_dataset.csv")

# =====================================
# Build Channel Statistics
# =====================================

channel_stats = (
    df.groupby("channel_title")
      .agg(
          channel_avg_views=("views", "mean"),
          channel_avg_likes=("likes", "mean"),
          channel_total_trending=("views", "count")
      )
      .reset_index()
)

# =====================================
# Category Dictionary
# =====================================

categories = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology"
}

# =====================================
# Streamlit UI starts here
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Analytics",
        "🤖 Prediction",
        "📈 Explainability",
        "ℹ️ About"
    ]
)

# ======================================================
# HOME
# ======================================================

if page == "🏠 Home":

    st.title("📈 Content Performance Predictor")

    st.caption(
        "An end-to-end Product Analytics dashboard that predicts high-engagement YouTube videos using Machine Learning."
    )

    st.markdown("---")

    st.markdown("""
    ## Project Overview

    This application predicts whether a YouTube video is likely to achieve **High Engagement**
    using Machine Learning.

    The project combines:

    - SQL Analytics
    - Exploratory Data Analysis
    - Feature Engineering
    - SentenceTransformer Embeddings
    - CatBoost
    - Explainable AI
    - Streamlit Dashboard
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Size",
            f"{len(df):,} Videos"
        )

    with col2:
        st.metric(
            "Best Model",
            "CatBoost"
        )

    with col3:
        st.metric(
            "ROC-AUC",
            "96.36%"
        )

    st.markdown("---")

    st.subheader("Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Total Channels**")
        st.metric("", df["channel_title"].nunique())

        st.write("**Categories**")
        st.metric("", df["category_id"].nunique())

    with col2:
        st.write("**Average Views**")
        st.metric("", f"{df['views'].mean():,.0f}")

        st.write("**Average Likes**")
        st.metric("", f"{df['likes'].mean():,.0f}")

# ======================================================
# ANALYTICS
# ======================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.write(
        "Exploratory Data Analysis performed on the dataset."
    )

    charts = [
        "views_distribution.png",
        "likes_distribution.png",
        "comments_distribution.png",
        "upload_hour_distribution.png",
        "upload_day_distribution.png",
        "category_distribution.png",
        "top_channels.png",
        "correlation_heatmap.png"
    ]

    tab1, tab2 = st.tabs(["EDA Charts", "Correlation"])

    with tab1:
        for chart in charts[:6]:
            path = os.path.join("outputs", chart)

            if os.path.exists(path):
                st.image(path, use_container_width=True)

    with tab2:
        for chart in charts[6:]:
            path = os.path.join("outputs", chart)

            if os.path.exists(path):
                st.image(path, use_container_width=True)

# ======================================================
# PREDICTION
# ======================================================

elif page == "🤖 Prediction":

    st.title("🤖 Predict Content Performance")

    title = st.text_input("Video Title")

    channel = st.selectbox(
        "Channel",
        sorted(df["channel_title"].unique())
    )

    # Get Channel Statistics
    stats = channel_stats[
        channel_stats["channel_title"] == channel
    ].iloc[0]

    avg_views = float(stats["channel_avg_views"])
    avg_likes = float(stats["channel_avg_likes"])
    total_trending = int(stats["channel_total_trending"])

    st.subheader("📊 Channel Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Average Views", f"{avg_views:,.0f}")
    c2.metric("Average Likes", f"{avg_likes:,.0f}")
    c3.metric("Trending Videos", total_trending)

    st.markdown("---")

    category = st.selectbox(
        "Category",
        list(categories.values())
    )

    raw_category = [
        k for k, v in categories.items()
        if v == category
    ][0]

    try:
        category_id = category_mapping.index(raw_category)
    except ValueError:
        st.error("Unknown category.")
        st.stop()

    publish_hour = st.slider(
        "Upload Hour",
        0,
        23,
        18
    )

    weekend = st.checkbox("Weekend Upload")

    if st.button("Predict"):

        if not title.strip():
            st.error("Please enter a video title before predicting.")
            st.stop()

        # Generate embedding
        embedding = embedding_model.encode(title)

        sample = {}

        # 384 embedding features
        for i in range(384):
            sample[f"title_emb_{i}"] = float(embedding[i])

        # Metadata
        sample["publish_hour"] = publish_hour
        sample["category_id"] = category_id
        sample["channel_avg_views"] = avg_views
        sample["channel_avg_likes"] = avg_likes
        sample["channel_total_trending"] = total_trending
        sample["is_weekend"] = int(weekend)

        sample = pd.DataFrame([sample])

        # sklearn stores training-time columns as `feature_names_in_`
        # (NOT `feature_names_`, which is CatBoost's attribute name).
        # Direct column selection (vs. reindex) fails loudly with a
        # KeyError if something is still missing, instead of silently
        # introducing NaNs.
        expected = list(model.feature_names_)

        missing = [c for c in expected if c not in sample.columns]
        extra = [c for c in sample.columns if c not in expected]

        if missing:
            st.error(f"Missing features ({len(missing)}): {missing}")
            st.stop()

        if extra:
            st.warning(f"Extra features ({len(extra)}): {extra}")

        sample = sample.reindex(columns=expected)

        prediction = model.predict(sample)
        probability = model.predict_proba(sample)

        pred = int(prediction[0])
        conf = float(probability[0][1])

        st.markdown("---")

        if pred == 1:
            st.success("🎉 High Engagement Predicted")
        else:
            st.error("📉 Normal Engagement Predicted")

        st.metric("Confidence", f"{conf:.2%}")
        st.progress(conf)

        st.subheader("Business Recommendations")

        if publish_hour in range(16, 22):
            st.success("✔ Good upload timing")
        else:
            st.warning("⚠ Evening uploads generally perform better")

        if avg_views > 500000:
            st.success("✔ Strong historical channel performance")

        if weekend:
            st.info("✔ Weekend uploads often receive higher engagement")

        if pred == 1:
            st.success("✅ Recommended for Promotion")
        else:
            st.warning("⚠ Monitor before Promotion")


# ======================================================
# EXPLAINABILITY
# ======================================================

elif page == "📈 Explainability":

    st.title("📈 Model Explainability")

    feature_path = "outputs/feature_importance.png"

    shap_path = "outputs/shap_summary.png"

    if os.path.exists(feature_path):

        st.subheader("Feature Importance")

        st.image(feature_path, use_container_width=True)

    else:

        st.warning("Run explain.py first.")

    if os.path.exists(shap_path):

        st.subheader("SHAP Summary")

        st.image(shap_path, use_container_width=True)


# ======================================================
# ABOUT
# ======================================================

else:

    st.title("About")

    st.markdown("""

## Content Performance Predictor

### Objective

Predict whether a YouTube video is likely to receive high engagement.

### Technologies

- Python
- Pandas
- SQLite
- Scikit-Learn
- Random Forest
- CatBoost
- SHAP
- Streamlit

### Workflow

- Data Cleaning
- SQL Analytics
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Explainability
- Dashboard Development

### Best Model

- Random Forest

Accuracy: **94.5%**

ROC-AUC: **0.96**

F1 Score: **0.86**

""")