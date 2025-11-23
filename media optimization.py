import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use("seaborn-v0_8")

df = pd.read_csv("mmm_outputs.csv")

channels = ["facebook", "search", "youtube", "email"]

def adstock(values, decay=0.5):
    x = np.zeros(len(values))
    for t in range(len(values)):
        if t == 0:
            x[t] = values[t]
        else:
            x[t] = values[t] + decay * x[t - 1]
    return x

def transform_media(spend, decay=0.5):
    adstock_vals = adstock(spend, decay)
    return np.log1p(adstock_vals)

import statsmodels.api as sm

# Rebuild X for model so we can reload the model parameters
X_cols = [ch + "_dr" for ch in channels]
X = df[X_cols]
X = sm.add_constant(X)

y = df["revenue"]

# Fit the same OLS model again (safe because all data is same)
model = sm.OLS(y, X).fit()
model.params


def predict_revenue_from_spend(spends, decay=0.5):
    transformed = []

    for spend, ch in zip(spends, channels):
        # Simulate a single spend level repeated (adstock needs a sequence)
        spend_series = pd.Series([spend])
        val = transform_media(spend_series, decay)[0]
        transformed.append(val)

    transformed = np.array(transformed)

    # β0 + Σ(β_i * channel_i)
    revenue_pred = model.params["const"] + np.sum(
        model.params[[ch + "_dr" for ch in channels]].values * transformed
    )

    return revenue_pred


def objective(spends):
    return -predict_revenue_from_spend(spends)

current_total_spend = df[channels].mean().sum()
budget = current_total_spend

budget

constraints = ({
    "type": "eq",
    "fun": lambda spends: np.sum(spends) - budget
})

bounds = [(0, None) for _ in channels]

initial_spend = np.ones(len(channels)) * (budget / len(channels))
initial_spend

result = minimize(
    objective,
    initial_spend,
    bounds=bounds,
    constraints=constraints,
    method="SLSQP",
)

resultresult = minimize(
    objective,
    initial_spend,
    bounds=bounds,
    constraints=constraints,
    method="SLSQP",
)

result

optimized_spend = result.x
optimized_spend


comparison = pd.DataFrame({
    "channel": channels,
    "current_avg_spend": df[channels].mean().values,
    "optimal_spend": optimized_spend
})

comparison["difference"] = comparison["optimal_spend"] - comparison["current_avg_spend"]

comparison

plt.figure(figsize=(10,6))
plt.bar(comparison["channel"], comparison["current_avg_spend"], alpha=0.5, label="Current")
plt.bar(comparison["channel"], comparison["optimal_spend"], alpha=0.8, label="Optimized")
plt.title("Current vs. Optimized Spend Allocation")
plt.ylabel("Spend")
plt.legend()
plt.show()

current_pred = predict_revenue_from_spend(df[channels].mean().values)
optimized_pred = predict_revenue_from_spend(optimized_spend)

print("Current predicted revenue:", current_pred)
print("Optimized predicted revenue:", optimized_pred)
print("Lift (%):", (optimized_pred - current_pred) / current_pred * 100)
