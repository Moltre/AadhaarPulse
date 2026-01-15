import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------- CONFIG ----------
# Default file in your repo
DATA_PATH = "data/biometric_updates.csv"

# SET THESE NAMES TO MATCH YOUR CSV HEADER (after lowercasing)
DATE_COL = "date"            # column that has dates
STATE_COL = "state"          # column with state names (or leave as "" if none)
DISTRICT_COL = "district"    # column with district names (or leave as "" if none)
VALUE_COL = "demo_age_5_17"  # numeric column you want to chart (e.g. demo_age_5_17)


# ---------- DATA LOADING ----------

@st.cache_data(ttl=300)
def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # show columns in sidebar so you can debug
    st.sidebar.write("Detected columns:", list(df.columns))

    # basic checks
    if DATE_COL.lower() not in df.columns:
        st.error(f"DATE_COL '{DATE_COL}' not found in CSV.")
        return pd.DataFrame()

    if VALUE_COL.lower() not in df.columns:
        st.error(f"VALUE_COL '{VALUE_COL}' not found in CSV.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[DATE_COL.lower()], errors="coerce")
    df = df.dropna(subset=["date"])

    df["metric"] = pd.to_numeric(df[VALUE_COL.lower()], errors="coerce").fillna(0)

    # optional columns
    if STATE_COL and STATE_COL.lower() in df.columns:
        df["state"] = df[STATE_COL.lower()].astype(str)
    else:
        df["state"] = "All"

    if DISTRICT_COL and DISTRICT_COL.lower() in df.columns:
        df["district"] = df[DISTRICT_COL.lower()].astype(str)
    else:
        df["district"] = "All"

    return df


# ---------- HELPERS ----------

def filter_data(df: pd.DataFrame, state: str, district: str, date_range):
    start_date, end_date = date_range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    if "state" in df.columns and state != "All":
        mask &= df["state"] == state
    if "district" in df.columns and district != "All":
        mask &= df["district"] == district

    return df[mask]


def kpi_block(df: pd.DataFrame):
    if df.empty:
        st.warning("No data for selected filters.")
        return

    total_val = float(df["metric"].sum())
    today = df["date"].max()
    yesterday = today - timedelta(days=1)

    today_val = float(df.loc[df["date"] == today, "metric"].sum())
    yesterday_val = float(df.loc[df["date"] == yesterday, "metric"].sum())

    col1, col2 = st.columns(2)
    col1.metric("Total value", f"{total_val:,.0f}")
    col2.metric("Today", f"{today_val:,.0f}", delta=today_val - yesterday_val)


def time_series_chart(df: pd.DataFrame):
    if df.empty:
        return
    daily = df.groupby("date", as_index=False)["metric"].sum()
    fig = px.line(daily, x="date", y="metric", title="Metric over time")
    st.plotly_chart(fig, use_container_width=True)


def state_bar_chart(df: pd.DataFrame):
    if df.empty or "state" not in df.columns:
        return
    s = (
        df.groupby("state", as_index=False)["metric"]
        .sum()
        .sort_values("metric", ascending=False)
        .head(10)
    )
    fig = px.bar(s, x="state", y="metric", title="Top 10 states")
    st.plotly_chart(fig, use_container_width=True)


def district_bar_chart(df: pd.DataFrame):
    if df.empty or "district" not in df.columns:
        return
    d = (
        df.groupby("district", as_index=False)["metric"]
        .sum()
        .sort_values("metric", ascending=False)
        .head(10)
    )
    fig = px.bar(d, x="district", y="metric", title="Top 10 districts")
    st.plotly_chart(fig, use_container_width=True)


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Aadhaar Real-time Monitor", layout="wide")

st.title("Aadhaar Real-time Monitor")
st.caption("Dashboard built from your Aadhaar aggregate CSV (flexible columns).")

st.sidebar.header("Data")

if not os.path.exists(DATA_PATH):
    st.sidebar.warning(
        f"Default data file not found: {DATA_PATH}. "
        "Upload a CSV/TXT using the uploader below."
    )

uploaded = st.sidebar.file_uploader(
    "Upload CSV/TXT", type=["csv", "txt"],
    help="Upload your Aadhaar aggregate file exported as CSV or TXT."
)

if uploaded is not None:
    df_raw = load_data(uploaded)
elif os.path.exists(DATA_PATH):
    df_raw = load_data(DATA_PATH)
else:
    df_raw = pd.DataFrame()

if df_raw.empty:
    st.stop()

min_date = df_raw["date"].min()
max_date = df_raw["date"].max()

state_options = ["All"] + sorted(df_raw["state"].dropna().unique().tolist())
district_options = ["All"] + sorted(df_raw["district"].dropna().unique().tolist())

state = st.sidebar.selectbox("State", state_options)
district = st.sidebar.selectbox("District", district_options)

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

df = filter_data(
    df_raw,
    state=state,
    district=district,
    date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date)),
)

kpi_block(df)

col_left, col_right = st.columns([2, 1])

with col_left:
    time_series_chart(df)
    state_bar_chart(df)

with col_right:
    district_bar_chart(df)

st.subheader("Filtered data (preview)")
st.dataframe(df.head(200))

st.caption(
    "If something looks wrong, open data/biometric_updates.csv on GitHub, "
    "see the column names, and update DATE_COL / VALUE_COL at the top."
)
