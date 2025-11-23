import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("deep")

raw_path = "data/raw/"
proc_path = "data/processed/"

events = pd.read_csv(raw_path + "user_events.csv")
events["event_time"] = pd.to_datetime(events["event_time"])

events.head()

print(events["event_type"].value_counts())
print(events["channel"].value_counts())
events.sort_values(["user_id", "event_time"]).head(10)



# First purchase time per user
first_purchase = (
    events[events["event_type"] == "purchase"]
    .groupby("user_id", as_index=False)["event_time"]
    .min()
    .rename(columns={"event_time": "first_purchase_time"})
)

len(first_purchase), first_purchase.head()

# Merge first purchase time into events
events_merged = events.merge(first_purchase, on="user_id", how="left")

# Keep only converting users
conv_events = events_merged[~events_merged["first_purchase_time"].isna()].copy()

# Keep events up to first purchase
conv_events = conv_events[
    conv_events["event_time"] <= conv_events["first_purchase_time"]
].copy()

conv_events.sort_values(["user_id", "event_time"]).head(15)

def build_channel_path(group):
    # Sort by time
    group = group.sort_values("event_time")
    # Keep only events that are touchpoints (we can include purchase channel as last)
    channels = group["channel"].tolist()
    return channels

paths = (
    conv_events
    .groupby("user_id")
    .apply(build_channel_path)
    .reset_index(name="path")
)

paths.head()
print("Example path:", paths["path"].iloc[0])
print("Number of converting users:", len(paths))


from collections import Counter

def attrib_last_touch(path):
    """100% credit to the last channel."""
    if not path:
        return {}
    last = path[-1]
    return {last: 1.0}

def attrib_first_touch(path):
    """100% credit to the first channel."""
    if not path:
        return {}
    first = path[0]
    return {first: 1.0}

def attrib_linear(path):
    """Equal credit to all touchpoints in the path."""
    if not path:
        return {}
    n = len(path)
    credit = 1.0 / n
    counts = Counter(path)
    # If the same channel appears multiple times, it accumulates
    return {ch: count * credit for ch, count in counts.items()}

def attrib_position_based_40_40_20(path):
    """
    40% to first, 40% to last, 20% distributed equally across middle touches.
    If only 1 touch → 100% to that.
    If 2 touches → 50/50.
    """
    if not path:
        return {}
    if len(path) == 1:
        return {path[0]: 1.0}
    if len(path) == 2:
        # Split evenly
        return {path[0]: 0.5, path[1]: 0.5}
    
    first = path[0]
    last = path[-1]
    middle = path[1:-1]
    
    credits = Counter()
    credits[first] += 0.4
    credits[last] += 0.4
    
    if middle:
        mid_share = 0.2 / len(middle)
        for ch in middle:
            credits[ch] += mid_share
    
    return dict(credits)

channels = ["facebook", "search", "youtube", "email"]

methods = {
    "last_touch": attrib_last_touch,
    "first_touch": attrib_first_touch,
    "linear": attrib_linear,
    "position_40_40_20": attrib_position_based_40_40_20,
}

# Initialize counters
attrib_results = {
    name: Counter() for name in methods.keys()
}

# Loop over each converting user path
for _, row in paths.iterrows():
    path = row["path"]
    for name, func in methods.items():
        contrib = func(path)
        attrib_results[name].update(contrib)

attrib_results

rows = []

for method_name, counter in attrib_results.items():
    total = sum(counter.values())  # total credited conversions (should equal #conversions)
    for ch in channels:
        val = counter.get(ch, 0.0)
        share = val / total if total > 0 else 0.0
        rows.append({
            "method": method_name,
            "channel": ch,
            "conversions_attributed": val,
            "share": share,
        })

attrib_df = pd.DataFrame(rows)
attrib_df

plt.figure(figsize=(10,6))
sns.barplot(
    data=attrib_df,
    x="channel",
    y="share",
    hue="method"
)
plt.title("Channel Attribution Shares by Method")
plt.ylabel("Share of Attributed Conversions")
plt.show()

channels = ["facebook", "search", "youtube", "email"]

# Load MMM contributions
mmm_contrib = pd.read_csv("channel_contrib.csv", header=None, names=["channel", "contribution"])

# Drop the NaN row
mmm_contrib = mmm_contrib.dropna(subset=["channel"])

# Strip the "_contrib" suffix to get clean channel names
mmm_contrib["channel"] = mmm_contrib["channel"].str.replace("_contrib", "", regex=False)

# Set index to clean channel name
mmm_contrib = mmm_contrib.set_index("channel")["contribution"]

# Normalize to shares and align with our channels list
mmm_share = (mmm_contrib / mmm_contrib.sum()).reindex(channels)

mmm_share

# Pick one MTA method to compare (e.g., last_touch)
last_touch_shares = (
    attrib_df[attrib_df["method"] == "last_touch"]
    .set_index("channel")["share"]
    .reindex(channels)
)

comparison = pd.DataFrame({
    "MMM_share": mmm_share,
    "MTA_last_touch_share": last_touch_shares,
})

comparison

comparison.plot(kind="bar", figsize=(10,6))
plt.title("Channel Share: MMM vs Last-Touch MTA")
plt.ylabel("Share")
plt.xticks(rotation=0)
plt.show()

attrib_df.to_csv("mta_attribution_results.csv", index=False)
comparison.to_csv("mmm_vs_mta_comparison.csv")
print("Saved MTA and MMM vs MTA comparison results.")
