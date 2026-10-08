import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
input_file = BASE_DIR / "data" / "household_power_consumption.txt"
output_file = BASE_DIR / "data" / "energy_15min.csv"

# Read dataset
df = pd.read_csv(
    input_file,
    sep=";",
    low_memory=False,
    na_values="?"
)

print("Original shape:", df.shape)

# Create datetime column
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

# Convert power column to numeric
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

# Keep only required columns
df = df[["datetime", "Global_active_power"]]

# Sort by datetime
df = df.sort_values("datetime")

# Set datetime as index
df = df.set_index("datetime")

# Fill missing values
df["Global_active_power"] = (
    df["Global_active_power"].interpolate()
)

print("\nMissing values after interpolation:")
print(df.isnull().sum())

# Resample to 15-minute intervals
df_15min = df.resample("15min").mean()

# Fill missing 15-minute values
df_15min["Global_active_power"] = (
    df_15min["Global_active_power"].interpolate()
)

# Remove any remaining missing values
df_15min = df_15min.dropna()

print("\n15-minute data:")
print(df_15min.head())

print("\n15-minute shape:")
print(df_15min.shape)

# Save processed dataset
df_15min.to_csv(output_file)

print("\nProcessed dataset saved successfully!")
print("Saved to:", output_file)