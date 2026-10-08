import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# File paths
data_path = BASE_DIR / "data" / "energy_15min.csv"
model_dir = BASE_DIR / "model"

model_dir.mkdir(exist_ok=True)

scaler_path = model_dir / "energy_scaler.pkl"


# Load processed data
df = pd.read_csv(
    data_path,
    parse_dates=["datetime"],
    index_col="datetime"
)

data = df[["Global_active_power"]].values


# Same 80/20 split used during LSTM training
train_size = int(len(data) * 0.80)

train_data = data[:train_size]


# Fit scaler ONLY on training data
scaler = MinMaxScaler()

scaler.fit(train_data)


# Save scaler
joblib.dump(
    scaler,
    scaler_path
)


print("Scaler created successfully!")
print("Scaler saved at:")
print(scaler_path)