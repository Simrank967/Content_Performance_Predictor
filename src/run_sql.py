import sqlite3
import pandas as pd

DATABASE = "youtube.db"

def run_query(sql_file):
    conn = sqlite3.connect(DATABASE)

    with open(sql_file, "r") as f:
        query = f.read()

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


queries = {
    "Upload Hour": "sql/best_upload_hour.sql",
    "Upload Day": "sql/best_upload_day.sql",
    "Category": "sql/category_analysis.sql",
    "Channel": "sql/channel_analysis.sql",
    "Engagement": "sql/engagement_analysis.sql",
    "Title": "sql/title_analysis.sql"
}

for name, file in queries.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    df = run_query(file)

    print(df.head(10))