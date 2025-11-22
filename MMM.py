import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

plt.style.use("seaborn-v0_8")
sns.set_palette("deep")

raw_path = "data/raw/"

ad = pd.read_csv(raw_path + "ad_spend_by_channel.csv")
sales = pd.read_csv(raw_path + "daily_sales.csv")

ad["date"] = pd.to_datetime(ad["date"])
sales["date"] = pd.to_datetime(sales["date"])

daily = ad.pivot_table(
    index="date",
    columns="channel",
    values="spend",
    aggfunc="sum"
).reset_index()

daily.columns.name = None
daily.head()

df = pd.merge(daily, sales, on="date", how="inner")
df.head()

def adstock(series, decay):
    x = np.zeros(len(series))
    for t in range(len(series)):
        if t == 0:
            x[t] = series.iloc[t]
        else:
            x[t] = series.iloc[t] + decay * x[t-1]
    return x

decay = 0.5  # you can tune this later

for ch in ["facebook", "search", "youtube", "email"]:
    df[ch + "_adstock"] = adstock(df[ch], decay)

for ch in ["facebook", "search", "youtube", "email"]:
    df[ch + "_dr"] = np.log1p(df[ch + "_adstock"])

y = df["revenue"]
X = df[[ch + "_dr" for ch in ["facebook", "search", "youtube", "email"]]]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
model.summary()

df["y_pred"] = model.predict(X)

for ch in ["facebook", "search", "youtube", "email"]:
    df[ch + "_contrib"] = model.params[ch + "_dr"] * df[ch + "_dr"]


channel_contrib = df[[ch + "_contrib" for ch in ["facebook", "search", "youtube", "email"]]].sum()
channel_contrib

channel_contrib.sort_values().plot(kind="barh", figsize=(7,4))
plt.title("Total Channel Contribution to Revenue")
plt.show()

# List of channels used in the model
channels = ["facebook", "search", "youtube", "email"]

# 1) Total spend per channel over the whole period
total_spend = (
    ad.groupby("channel")["spend"]
      .sum()
      .reindex(channels)
)

# 2) Total incremental revenue contribution per channel from MMM
channel_contrib = pd.Series(
    {ch: df[f"{ch}_contrib"].sum() for ch in channels}
)

print("Total spend:")
print(total_spend)
print("\nChannel contribution (incremental revenue):")
print(channel_contrib)

# 3) ROI = incremental revenue / spend
roi = channel_contrib / total_spend

print("\nROI (incremental revenue per $1 spent):")
print(roi)

# 4) Plot
roi.sort_values().plot(kind="barh", figsize=(7,4))
plt.title("ROI per Channel")
plt.xlabel("Incremental Revenue per $1 Spent")
plt.show()

response_curves = {}

for ch in ["facebook", "search", "youtube", "email"]:
    spend_range = np.linspace(0, df[ch].max()*1.5, 100)
    adstock_vals = adstock(pd.Series(spend_range), decay)
    dr_vals = np.log1p(adstock_vals)
    
    # predicted incremental
    pred = model.params[ch + "_dr"] * dr_vals
    
    response_curves[ch] = (spend_range, pred)


plt.figure(figsize=(10,6))
for ch, (spend_range, pred) in response_curves.items():
    plt.plot(spend_range, pred, label=ch)
plt.legend()
plt.title("Response Curves by Channel")
plt.xlabel("Spend")
plt.ylabel("Incremental Revenue")
plt.show()

df.to_csv("mmm_outputs.csv", index=False)
roi.to_csv("channel_roi.csv")
channel_contrib.to_csv("channel_contrib.csv")

print("MMM results saved.")

