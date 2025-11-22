import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("deep")

raw_path = "data/raw/"

ad = pd.read_csv(raw_path + "ad_spend_by_channel.csv")
sales = pd.read_csv(raw_path + "daily_sales.csv")
events = pd.read_csv(raw_path + "user_events.csv")
crm = pd.read_csv(raw_path + "crm_customers.csv")

ad.head(), sales.head(), events.head(), crm.head()

ad["date"] = pd.to_datetime(ad["date"])
sales["date"] = pd.to_datetime(sales["date"])
events["event_time"] = pd.to_datetime(events["event_time"])
crm["signup_date"] = pd.to_datetime(crm["signup_date"])

print("Ad rows:", len(ad))
print("Sales rows:", len(sales))
print("Events rows:", len(events))
print("CRM rows:", len(crm))

print("\nDate ranges:")
print("ad:", ad["date"].min(), "to", ad["date"].max())
print("sales:", sales["date"].min(), "to", sales["date"].max())
print("events:", events["event_time"].min(), "to", events["event_time"].max())
print("crm:", crm["signup_date"].min(), "to", crm["signup_date"].max())

plt.figure(figsize=(12,5))
plt.plot(sales["date"], sales["revenue"])
plt.title("Daily Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(12,5))
for ch in ad["channel"].unique():
    plt.plot(
        ad[ad["channel"] == ch]["date"],
        ad[ad["channel"] == ch]["spend"],
        label=ch
    )
plt.title("Daily Spend by Channel")
plt.legend()
plt.show()

conv_by_channel = ad.groupby("channel")["conversions"].sum().sort_values(ascending=False)
conv_by_channel


conv_by_channel.plot(kind="bar", figsize=(7,4))
plt.title("Total Conversions by Channel")
plt.xlabel("Channel")
plt.ylabel("Conversions")
plt.show()

events["event_type"].value_counts()

events["date"] = events["event_time"].dt.date
pivot = events.pivot_table(
    index="date",
    columns="event_type",
    values="user_id",
    aggfunc="count"
)

plt.figure(figsize=(12,5))
sns.heatmap(pivot.fillna(0), cmap="Blues")
plt.title("Daily Events Heatmap")
plt.show()
