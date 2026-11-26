"""
Week 2: Exploratory Data Analysis of Bike-Sharing Transportation Demand
Run from the Week-02-EDA-Logistics folder:
    python Code/eda_analysis.py
Dependencies: pandas, numpy, matplotlib, seaborn
Input: Dataset/hour.csv and Dataset/day.csv
Outputs: Visualizations/*.png and Output/eda_summary.txt
Note: This is a public bike-sharing mobility dataset, not freight shipment data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
PLOTS = ROOT / "Visualizations"
OUT = ROOT / "Output"
PLOTS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

hour = pd.read_csv(DATA / "hour.csv")
day = pd.read_csv(DATA / "day.csv")
hour["dteday"] = pd.to_datetime(hour["dteday"], errors="coerce")
day["dteday"] = pd.to_datetime(day["dteday"], errors="coerce")

season_map = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
weather_map = {1:"Clear / partly cloudy", 2:"Mist / cloudy",
               3:"Light rain or snow", 4:"Heavy rain / snow"}
hour["season_name"] = hour["season"].map(season_map)
hour["weather_name"] = hour["weathersit"].map(weather_map)

# 1. Initial profiling and data-quality checks
summary = []
summary.append(f"Hourly rows: {len(hour):,}; columns: {hour.shape[1]}")
summary.append(f"Daily rows: {len(day):,}; columns: {day.shape[1]}")
summary.append(f"Hourly date range: {hour.dteday.min().date()} to {hour.dteday.max().date()}")
summary.append(f"Missing cells (hourly): {int(hour.isna().sum().sum())}")
summary.append(f"Duplicate rows (hourly): {int(hour.duplicated().sum())}")
summary.append(f"Total rentals (hourly): {int(hour.cnt.sum()):,}")
summary.append(f"Mean hourly rentals: {hour.cnt.mean():.2f}")
summary.append(f"Median hourly rentals: {hour.cnt.median():.2f}")
summary.append("\\nDescriptive statistics for key measures:")
summary.append(hour[["temp","atemp","hum","windspeed","casual","registered","cnt"]].describe().round(2).to_string())

# 2. Feature interpretation
hour["temperature_c"] = hour["temp"] * 41
hour["feels_like_c"] = hour["atemp"] * 50
hour["humidity_pct"] = hour["hum"] * 100
hour["windspeed_kmh_approx"] = hour["windspeed"] * 67

# 3. Visualizations
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,5))
sns.lineplot(data=hour.groupby("hr", as_index=False)["cnt"].mean(), x="hr", y="cnt", marker="o")
plt.title("Average Bike Rentals by Hour"); plt.xlabel("Hour"); plt.ylabel("Mean rentals")
plt.xticks(range(24)); plt.tight_layout(); plt.savefig(PLOTS/"01_hourly_demand.png", dpi=160); plt.close()

plt.figure(figsize=(10,5))
sns.lineplot(data=hour, x="hr", y="cnt", hue="workingday", estimator="mean", errorbar=None)
plt.title("Hourly Demand: Working vs Non-working Days")
plt.xlabel("Hour"); plt.ylabel("Mean rentals"); plt.tight_layout()
plt.savefig(PLOTS/"02_workday_profile.png", dpi=160); plt.close()

plt.figure(figsize=(9,5))
sns.boxplot(data=hour, x="season_name", y="cnt", showfliers=False,
            order=["Spring","Summer","Fall","Winter"])
plt.title("Hourly Rental Distribution by Season"); plt.xlabel("Season"); plt.ylabel("Rentals")
plt.tight_layout(); plt.savefig(PLOTS/"03_season_boxplot.png", dpi=160); plt.close()

plt.figure(figsize=(10,5))
weather_avg = hour.groupby("weather_name").cnt.mean().sort_values(ascending=False)
plt.bar(weather_avg.index, weather_avg.values)
plt.title("Average Rentals by Weather"); plt.xlabel("Weather condition"); plt.ylabel("Mean rentals")
plt.xticks(rotation=18, ha="right"); plt.tight_layout(); plt.savefig(PLOTS/"04_weather_bar.png", dpi=160); plt.close()

plt.figure(figsize=(9,5))
plt.hist(hour["cnt"], bins=40)
plt.title("Distribution of Hourly Rental Counts"); plt.xlabel("Rentals per hour")
plt.tight_layout(); plt.savefig(PLOTS/"05_demand_histogram.png", dpi=160); plt.close()

plt.figure(figsize=(9,5))
sns.scatterplot(data=hour.sample(min(2500,len(hour)), random_state=42),
                x=hour.sample(min(2500,len(hour)), random_state=42)["temp"]*41,
                y=hour.sample(min(2500,len(hour)), random_state=42)["cnt"], alpha=.35, s=18)
plt.title("Temperature vs Hourly Rentals"); plt.xlabel("Temperature (°C)"); plt.ylabel("Rentals")
plt.tight_layout(); plt.savefig(PLOTS/"06_temperature_scatter.png", dpi=160); plt.close()

corr = hour[["temp","atemp","hum","windspeed","casual","registered","cnt"]].corr()
plt.figure(figsize=(9,7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix"); plt.tight_layout()
plt.savefig(PLOTS/"07_correlation_heatmap.png", dpi=160); plt.close()

monthly = hour.assign(month_period=hour.dteday.dt.to_period("M")).groupby("month_period").cnt.sum()
plt.figure(figsize=(12,5))
monthly.plot(marker="o")
plt.title("Monthly Total Rentals"); plt.xlabel("Month"); plt.ylabel("Total rentals")
plt.tight_layout(); plt.savefig(PLOTS/"08_monthly_trend.png", dpi=160); plt.close()

user_totals = hour[["registered","casual"]].sum().rename(index={"registered":"Registered","casual":"Casual"})
plt.figure(figsize=(7,5)); plt.bar(user_totals.index, user_totals.values)
plt.title("Total Rentals by User Type"); plt.ylabel("Total rentals")
plt.tight_layout(); plt.savefig(PLOTS/"09_user_type.png", dpi=160); plt.close()

weekday_names = {0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"}
hour["weekday_name"] = hour["weekday"].map(weekday_names)
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_avg = hour.groupby("weekday_name").cnt.mean().reindex(weekday_order)
plt.figure(figsize=(9,5)); plt.bar(weekday_avg.index, weekday_avg.values)
plt.title("Average Hourly Rentals by Day of Week"); plt.ylabel("Mean rentals")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(PLOTS/"10_weekday.png", dpi=160); plt.close()

summary.append("\\nMean rentals by hour:")
summary.append(hour.groupby("hr").cnt.mean().round(2).to_string())
summary.append("\\nMean rentals by season:")
summary.append(hour.groupby("season_name").cnt.mean().round(2).to_string())
summary.append("\\nMean rentals by weather:")
summary.append(hour.groupby("weather_name").cnt.mean().round(2).to_string())
summary.append("\\nCorrelation with total rentals:")
summary.append(hour[["temp","atemp","hum","windspeed","casual","registered","cnt"]].corr()["cnt"].round(3).to_string())

(OUT/"eda_summary.txt").write_text("\\n".join(summary), encoding="utf-8")
print("\\n".join(summary[:9]))
print(f"\\nCreated 10 visualizations in: {PLOTS}")
print(f"Summary saved to: {OUT/'eda_summary.txt'}")
