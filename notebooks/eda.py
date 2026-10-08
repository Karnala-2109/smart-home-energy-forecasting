import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Processed dataset
file_path = BASE_DIR / "data" / "energy_15min.csv"

# Read data
df = pd.read_csv(
    file_path,
    parse_dates=["datetime"],
    index_col="datetime"
)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 1. Overall Energy Consumption
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(
    df.index,
    df["Global_active_power"]
)

plt.title("Overall Household Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. One Week Consumption
# --------------------------------------------------

one_week = df.iloc[:7 * 24 * 4]

plt.figure(figsize=(15, 5))

plt.plot(
    one_week.index,
    one_week["Global_active_power"]
)

plt.title("Household Energy Consumption - First Week")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Daily Average Consumption
# --------------------------------------------------

daily_average = df["Global_active_power"].resample("D").mean()

plt.figure(figsize=(15, 5))

plt.plot(
    daily_average.index,
    daily_average
)

plt.title("Daily Average Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Average Global Active Power (kW)")
plt.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Hourly Consumption Pattern
# --------------------------------------------------

hourly_average = (
    df.groupby(df.index.hour)["Global_active_power"]
    .mean()
)

plt.figure(figsize=(12, 5))

plt.plot(
    hourly_average.index,
    hourly_average.values,
    marker="o"
)

plt.title("Average Energy Consumption by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Global Active Power (kW)")
plt.xticks(range(24))
plt.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Consumption Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["Global_active_power"],
    bins=50
)

plt.title("Distribution of Household Energy Consumption")
plt.xlabel("Global Active Power (kW)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Peak Usage
# --------------------------------------------------

peak_value = df["Global_active_power"].max()
peak_time = df["Global_active_power"].idxmax()

print("\nPeak Energy Consumption:")
print("Power:", peak_value, "kW")
print("Time:", peak_time)


# --------------------------------------------------
# 7. Average Consumption
# --------------------------------------------------

average_value = df["Global_active_power"].mean()

print("\nAverage Energy Consumption:")
print(average_value, "kW")