import streamlit as st
import pandas as pd
import sqlite3
from db import db
from db import DB_PATH


st.set_page_config(
    page_title="Travel Agent Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Global Wanderer Analytics")

# =================================
# Metrics
# =================================

stats = db.get_stats()
st.write("DB File Exists:", DB_PATH.exists())
st.write("DB Path:", DB_PATH)

import os
st.write("DB Size:", os.path.getsize(DB_PATH))
c1, c2, c3, c4 = st.columns(4)

c1.metric("Sessions", stats["sessions"])
c2.metric("Messages", stats["messages"])
c3.metric("Tool Calls", stats["tool_calls"])
c4.metric("Errors", stats["errors"])

st.divider()

# =================================
# Tool Usage
# =================================

st.subheader("🛠 Tool Usage")

tool_df = pd.DataFrame(db.get_tool_usage())

if not tool_df.empty:
    st.bar_chart(
        tool_df.set_index("tool_name")
    )

    st.dataframe(tool_df)

# =================================
# Top Cities
# =================================

st.subheader("🏙 Top Cities")

city_df = pd.DataFrame(db.get_top_cities())

if not city_df.empty:
    st.bar_chart(
        city_df.set_index("city")
    )

    st.dataframe(city_df)

# =================================
# Daily Activity
# =================================

st.subheader("📈 Daily Messages")

activity_df = pd.DataFrame(
    db.get_daily_activity()
)

if not activity_df.empty:
    st.line_chart(
        activity_df.set_index("day")
    )

    st.dataframe(activity_df)


#________________LOGTABLE______________________
conn = sqlite3.connect("travel_agent.db")

df = pd.read_sql_query(
    "SELECT * FROM tool_calls",
    conn
)

st.write("Number of rows:", len(df))
st.dataframe(df)

  

#_______________RESPONSE___________________________
st.subheader("⚡ Average Tool Response Time")

perf_df = pd.read_sql_query("""
SELECT
    tool_name,
    AVG(duration_ms) as avg_time
FROM tool_calls
GROUP BY tool_name
ORDER BY avg_time DESC
""", conn)

st.dataframe(perf_df)

if not perf_df.empty:
    st.bar_chart(
        perf_df.set_index("tool_name")
    )

st.subheader("📈 Daily Tool Use")

daily_df = pd.read_sql_query("""
SELECT
    DATE(called_at) as day,
    COUNT(*) as searches
FROM tool_calls
GROUP BY DATE(called_at)
ORDER BY day
""", conn)

st.dataframe(daily_df)

if not daily_df.empty:
    st.line_chart(
        daily_df.set_index("day")
    )

conn.close()      