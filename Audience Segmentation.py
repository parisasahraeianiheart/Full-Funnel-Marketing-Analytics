import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.style.use("seaborn-v0_8")
sns.set_palette("deep")

raw_path = "data/raw/"

events = pd.read_csv(raw_path + "user_events.csv")
crm = pd.read_csv(raw_path + "crm_customers.csv")

events["event_time"] = pd.to_datetime(events["event_time"])
crm["signup_date"] = pd.to_datetime(crm["signup_date"])

events.head(), crm.head()

# Total events by user
summary = (
    events.groupby("user_id")
          .agg(
              impressions=("event_type", lambda x: (x=="impression").sum()),
              clicks=("event_type", lambda x: (x=="click").sum()),
              purchases=("event_type", lambda x: (x=="purchase").sum()),
              first_event=("event_time", "min"),
              last_event=("event_time", "max"),
              unique_channels=("channel", "nunique"),
              total_events=("event_type", "count")
          )
          .reset_index()
)

summary["days_active"] = (summary["last_event"] - summary["first_event"]).dt.days + 1
summary["recency_days"] = (events["event_time"].max() - summary["last_event"]).dt.days
summary["ctr_ratio"] = summary["clicks"] / summary["impressions"].replace(0, np.nan)
summary["purchase_flag"] = (summary["purchases"] > 0).astype(int)

df = pd.merge(summary, crm, on="user_id", how="left")
df.head()

numeric_cols = [
    "impressions",
    "clicks",
    "purchases",
    "total_events",
    "unique_channels",
    "days_active",
    "recency_days",
    "ctr_ratio",
]

X = df[numeric_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K, inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)
df.head()

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)

df["pc1"] = pca_coords[:, 0]
df["pc2"] = pca_coords[:, 1]

plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="segment", palette="tab10")
plt.title("User Segments (PCA projection)")
plt.show()

segment_profiles = df.groupby("segment")[numeric_cols].mean()
segment_profiles

def label_segment(row):
    if row["segment"] == 0:
        return "Low Engagement"
    elif row["segment"] == 1:
        return "High-Value Buyers"
    elif row["segment"] == 2:
        return "Search-Heavy Explorers"
    elif row["segment"] == 3:
        return "Multi-Channel Browsers"
    else:
        return "Unknown"

df["segment_name"] = df.apply(label_segment, axis=1)

df[["user_id", "segment", "segment_name"]].head()

df.to_csv("user_segments.csv", index=False)
print("Saved user segments.")

