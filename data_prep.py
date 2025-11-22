"""
data_prep.py

Simulate synthetic marketing data for:
- Marketing Mix Modeling (MMM)
- Clean-room–style analysis (join CRM + platform logs on user_id)
- Audience segmentation
- Simple MTA-style path analysis
- Media optimization

Outputs (saved under data/raw/):
- ad_spend_by_channel.csv
- daily_sales.csv
- user_events.csv
- crm_customers.csv
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# -----------------------------
# Global config
# -----------------------------

SEED = 42
np.random.seed(SEED)

N_DAYS = 180
START_DATE = datetime(2024, 1, 1)

CHANNELS = ["facebook", "search", "youtube", "email"]

# Base daily spend per channel (in dollars)
BASE_SPEND = {
    "facebook": 4000,
    "search": 5000,
    "youtube": 2500,
    "email": 500,
}

# Approximate CPMs (cost per 1000 impressions)
CPM = {
    "facebook": 8,
    "search": 10,
    "youtube": 6,
    "email": 4,
}

# Approximate click-through rates
CTR = {
    "facebook": 0.015,
    "search": 0.03,
    "youtube": 0.008,
    "email": 0.05,
}

# Approximate conversion rates (post-click)
CVR = {
    "facebook": 0.03,
    "search": 0.05,
    "youtube": 0.02,
    "email": 0.04,
}

AVG_ORDER_VALUE = 80.0  # dollars

# Number of additional non-converting users to simulate
NON_CONVERTING_USER_MULTIPLIER = 3


# -----------------------------
# Helper functions
# -----------------------------

def make_date_range(n_days: int, start_date: datetime) -> pd.DatetimeIndex:
    """Create a daily date range."""
    return pd.date_range(start=start_date, periods=n_days, freq="D")


def seasonality_factor(day_index: int) -> float:
    """
    Simple weekly seasonality:
    - weekends slightly higher demand
    """
    # day_index 0=Monday if we start Monday, but we'll just mod by 7
    dow = day_index % 7
    if dow in [5, 6]:  # Saturday, Sunday
        return 1.15
    elif dow in [0]:  # Monday slump
        return 0.9
    else:
        return 1.0


# -----------------------------
# 1. Simulate daily channel-level spend + performance
# -----------------------------

def simulate_channel_daily_data() -> pd.DataFrame:
    """
    Simulate daily spend, impressions, clicks, conversions by channel.

    Returns:
        DataFrame with columns:
            date, channel, spend, impressions, clicks, conversions
    """
    dates = make_date_range(N_DAYS, START_DATE)
    rows = []

    for day_idx, date in enumerate(dates):
        seas = seasonality_factor(day_idx)

        for ch in CHANNELS:
            base = BASE_SPEND[ch] * seas

            # Add some random noise, but keep spend positive
            spend = max(
                100.0,
                np.random.normal(loc=base, scale=0.15 * base)
            )

            # Impressions from spend and CPM
            impressions = int((spend / CPM[ch]) * 1000)

            # Clicks from impressions and CTR
            clicks = np.random.binomial(
                n=impressions,
                p=min(CTR[ch], 0.3),  # cap probabilities
            )

            # Conversions from clicks and CVR
            conversions = np.random.binomial(
                n=clicks,
                p=min(CVR[ch], 0.5),
            )

            rows.append({
                "date": date,
                "channel": ch,
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            })

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# 2. Aggregate to daily sales
# -----------------------------

def build_daily_sales(ad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate conversions to daily orders and revenue.

    Returns:
        DataFrame with columns:
            date, orders, revenue
    """
    daily = (
        ad_df.groupby("date", as_index=False)["conversions"]
        .sum()
        .rename(columns={"conversions": "orders"})
    )

    # Revenue with some noise around average order value
    noise = np.random.normal(loc=1.0, scale=0.05, size=len(daily))
    daily["revenue"] = (daily["orders"] * AVG_ORDER_VALUE * noise).round(2)

    return daily


# -----------------------------
# 3. Simulate user-level events (for paths / MTA / segmentation)
# -----------------------------

def simulate_user_paths(ad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-level event paths based on conversions per day+channel.

    For each conversion we create:
    - 1-3 pre-conversion events (impressions/clicks across channels)
    - 1 purchase event

    Additionally, we simulate non-converting users who have impressions/clicks
    but no purchase.

    Returns:
        DataFrame with columns:
            user_id, event_time, event_type, channel
    """
    events = []
    user_id_counter = 1

    # Convert date to pure date (no time) for easier handling
    ad_df = ad_df.copy()
    ad_df["date"] = pd.to_datetime(ad_df["date"])

    # Converting users
    for _, row in ad_df.iterrows():
        date = row["date"]
        ch = row["channel"]
        n_conv = int(row["conversions"])

        for _ in range(n_conv):
            uid = user_id_counter
            user_id_counter += 1

            # Number of pre-conversion touches
            n_pre = np.random.randint(1, 4)  # 1-3

            # Random lookback days (within last 7 days)
            for _ in range(n_pre):
                lookback = np.random.randint(1, 8)
                event_date = date - timedelta(days=lookback)
                pre_channel = np.random.choice(CHANNELS)

                evt_type = np.random.choice(["impression", "click"], p=[0.7, 0.3])

                events.append({
                    "user_id": uid,
                    "event_time": event_date + timedelta(
                        hours=np.random.randint(8, 22)
                    ),
                    "event_type": evt_type,
                    "channel": pre_channel,
                })

            # Purchase event
            events.append({
                "user_id": uid,
                "event_time": date + timedelta(
                    hours=np.random.randint(8, 22)
                ),
                "event_type": "purchase",
                "channel": ch,  # last-touch channel
            })

    n_converting_users = user_id_counter - 1

    # Non-converting users
    n_non_conv_users = NON_CONVERTING_USER_MULTIPLIER * n_converting_users

    for _ in range(n_non_conv_users):
        uid = user_id_counter
        user_id_counter += 1

        # choose a random date in the range
        day_offset = np.random.randint(0, N_DAYS)
        base_date = START_DATE + timedelta(days=day_offset)

        # 1–4 events
        n_evts = np.random.randint(1, 5)
        for _ in range(n_evts):
            offset = np.random.randint(0, 3)
            event_date = base_date + timedelta(days=offset)
            ch = np.random.choice(CHANNELS)
            evt_type = np.random.choice(["impression", "click"], p=[0.8, 0.2])

            events.append({
                "user_id": uid,
                "event_time": event_date + timedelta(
                    hours=np.random.randint(8, 22)
                ),
                "event_type": evt_type,
                "channel": ch,
            })

    events_df = pd.DataFrame(events)
    events_df.sort_values(["user_id", "event_time"], inplace=True)
    events_df.reset_index(drop=True, inplace=True)

    return events_df


# -----------------------------
# 4. CRM / customer table
# -----------------------------

def build_crm_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a CRM-like table with one row per user.

    For each user:
    - signup_date: first event_time
    - primary_channel: most frequent channel
    - region: random region
    - ltv_bucket: based on whether user purchased

    Returns:
        DataFrame with columns:
            user_id, signup_date, primary_channel, region, ltv_bucket
    """
    # First event per user
    first_events = (
        events_df.sort_values("event_time")
        .groupby("user_id", as_index=False)
        .first()[["user_id", "event_time", "channel"]]
    )
    first_events.rename(
        columns={"event_time": "signup_date", "channel": "first_channel"},
        inplace=True,
    )

    # Primary channel: most frequent channel per user
    channel_counts = (
        events_df.groupby(["user_id", "channel"])
        .size()
        .reset_index(name="cnt")
    )
    primary_ch = (
        channel_counts.sort_values(["user_id", "cnt"], ascending=[True, False])
        .drop_duplicates("user_id")[["user_id", "channel"]]
        .rename(columns={"channel": "primary_channel"})
    )

    crm = pd.merge(first_events, primary_ch, on="user_id", how="left")

    # Did user ever purchase?
    purchases = (
        events_df[events_df["event_type"] == "purchase"]
        .groupby("user_id")
        .size()
        .reset_index(name="n_purchases")
    )

    crm = pd.merge(crm, purchases, on="user_id", how="left")
    crm["n_purchases"] = crm["n_purchases"].fillna(0).astype(int)

    # Assign regions
    regions = ["NA", "EU", "APAC", "LATAM"]
    crm["region"] = np.random.choice(regions, size=len(crm), p=[0.5, 0.2, 0.2, 0.1])

    # LTV bucket based on purchases + random noise
    def assign_ltv(row):
        if row["n_purchases"] == 0:
            return "low"
        elif row["n_purchases"] == 1:
            return np.random.choice(["medium", "high"], p=[0.7, 0.3])
        else:
            return "high"

    crm["ltv_bucket"] = crm.apply(assign_ltv, axis=1)

    # Tidy columns
    crm["signup_date"] = crm["signup_date"].dt.date

    return crm[["user_id", "signup_date", "primary_channel", "region", "ltv_bucket"]]


# -----------------------------
# Main orchestration
# -----------------------------

def main():
    # Ensure output directory exists
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1) Channel-level daily data
    ad_df = simulate_channel_daily_data()
    ad_path = os.path.join(raw_dir, "ad_spend_by_channel.csv")
    ad_df.to_csv(ad_path, index=False)
    print(f"Saved: {ad_path}")

    # 2) Daily sales
    sales_df = build_daily_sales(ad_df)
    sales_path = os.path.join(raw_dir, "daily_sales.csv")
    sales_df.to_csv(sales_path, index=False)
    print(f"Saved: {sales_path}")

    # 3) User-level events
    events_df = simulate_user_paths(ad_df)
    events_df["event_time"] = events_df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    events_path = os.path.join(raw_dir, "user_events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved: {events_path}")

    # 4) CRM table
    crm_df = build_crm_table(events_df)
    crm_path = os.path.join(raw_dir, "crm_customers.csv")
    crm_df.to_csv(crm_path, index=False)
    print(f"Saved: {crm_path}")

    print("\nData simulation complete ✅")


if __name__ == "__main__":
    main()
