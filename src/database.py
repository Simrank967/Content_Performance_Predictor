import sqlite3
import pandas as pd

# -----------------------------
# Load Cleaned Dataset
# -----------------------------
df = pd.read_csv("data/cleaned_youtube.csv")

# -----------------------------
# Create SQLite Database
# -----------------------------
conn = sqlite3.connect("youtube.db")

# -----------------------------
# Store Data in SQLite
# -----------------------------
df.to_sql(
    "youtube_videos",
    conn,
    if_exists="replace",
    index=False
)

print("Database Created Successfully!")

# -----------------------------
# Verify Data
# -----------------------------
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM youtube_videos")

rows = cursor.fetchone()[0]

print(f"Total Rows: {rows}")

cursor.execute("PRAGMA table_info(youtube_videos)")

print("\nColumns:")

for column in cursor.fetchall():
    print(column[1])

conn.close()