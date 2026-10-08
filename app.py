import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Home Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "energy_15min.csv"
model_path = BASE_DIR / "model" / "energy_lstm.keras"
scaler_path = BASE_DIR / "model" / "energy_scaler.pkl"


# --------------------------------------------------
# 3. Load model, scaler and data
# --------------------------------------------------

@st.cache_resource
def load_lstm_model():

    return load_model(model_path)


@st.cache_resource
def load_scaler():

    return joblib.load(scaler_path)


@st.cache_data
def load_energy_data():

    return pd.read_csv(
        data_path,
        parse_dates=["datetime"],
        index_col="datetime"
    )


model = load_lstm_model()
scaler = load_scaler()
df = load_energy_data()


# --------------------------------------------------
# 4. Title
# --------------------------------------------------

st.title("⚡ Smart Home Energy Forecasting")

st.write(
    "LSTM-based household electricity consumption forecasting"
)


# --------------------------------------------------
# 5. Current consumption
# --------------------------------------------------

current_consumption = df["Global_active_power"].iloc[-1]

latest_time = df.index[-1]


# --------------------------------------------------
# 6. Daily consumption
# --------------------------------------------------

today = df.index[-1].date()

today_data = df[
    df.index.date == today
]

daily_consumption = (
    today_data["Global_active_power"].sum() * 0.25
)


# --------------------------------------------------
# 7. Peak consumption
# --------------------------------------------------

peak_consumption = df[
    "Global_active_power"
].max()


# --------------------------------------------------
# 8. Dashboard metrics
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric("Latest Recorded Consumption", f"{current_consumption:.2f} kW")

with col2:

   st.metric("Latest Day Consumption", f"{daily_consumption:.2f} kWh")


with col3:

    st.metric(
        "Peak Consumption",
        f"{peak_consumption:.2f} kW"
    )


st.divider()


# --------------------------------------------------
# 9. Prepare latest 24 hours
# --------------------------------------------------

sequence_length = 96

latest_data = df[
    ["Global_active_power"]
].values[-sequence_length:]


scaled_latest = scaler.transform(
    latest_data
)


# --------------------------------------------------
# 10. Generate future predictions
# --------------------------------------------------

future_predictions = []

current_sequence = scaled_latest.copy()


for step in range(4):

    input_data = current_sequence.reshape(
        1,
        sequence_length,
        1
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )[0][0]

    future_predictions.append(prediction)

    # Add prediction to sequence
    current_sequence = np.append(
        current_sequence[1:],
        [[prediction]],
        axis=0
    )


# Convert predictions back to kW

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
).flatten()


# --------------------------------------------------
# 11. Prediction section
# --------------------------------------------------

st.subheader("🔮 Future Energy Prediction")


pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)


with pred_col1:

    st.metric(
        "Next 15 Minutes",
        f"{future_predictions[0]:.2f} kW"
    )


with pred_col2:

    st.metric(
        "Next 30 Minutes",
        f"{future_predictions[1]:.2f} kW"
    )


with pred_col3:

    st.metric(
        "Next 45 Minutes",
        f"{future_predictions[2]:.2f} kW"
    )


with pred_col4:

    st.metric(
        "Next 60 Minutes",
        f"{future_predictions[3]:.2f} kW"
    )


# --------------------------------------------------
# 12. Future prediction graph
# --------------------------------------------------

prediction_times = ["15 min", "30 min", "45 min", "60 min"]

prediction_df = pd.DataFrame({
    "Forecast Horizon": prediction_times,
    "Predicted Power (kW)": future_predictions
})

st.dataframe(
    prediction_df,
    use_container_width=True,
    hide_index=True
)

st.line_chart(
    prediction_df.set_index("Forecast Horizon")
)

# --------------------------------------------------
# 13. Recent consumption graph
# --------------------------------------------------

st.subheader("📈 Recent Energy Consumption")

recent_data = df.tail(96)

st.line_chart(
    recent_data["Global_active_power"]
)

# Peak Usage Analysis
st.subheader("🔥 Peak Usage Analysis")
# Hourly Consumption Pattern
st.subheader("🕐 Hourly Consumption Pattern")

hourly_consumption = (
    df.groupby(df.index.hour)["Global_active_power"]
    .mean()
    .reset_index()
)

hourly_consumption.columns = [
    "Hour",
    "Average Power (kW)"
]

hourly_consumption["Hour"] = (
    hourly_consumption["Hour"].astype(str) + ":00"
)

st.bar_chart(
    hourly_consumption.set_index("Hour")
)

peak_data = df.nlargest(
    5,
    "Global_active_power"
)[["Global_active_power"]].copy()

peak_data["Time"] = peak_data.index.strftime("%Y-%m-%d %H:%M")

peak_data = peak_data[
    ["Time", "Global_active_power"]
].rename(
    columns={
        "Time": "Date & Time",
        "Global_active_power": "Power Consumption (kW)"
    }
)

st.dataframe(
    peak_data,
    use_container_width=True,
    hide_index=True
)
# --------------------------------------------------
# 14. High consumption warning
# --------------------------------------------------

st.subheader("⚠️ Energy Usage Status")

average_consumption = df[
    "Global_active_power"
].mean()

next_prediction = future_predictions[0]


if next_prediction > average_consumption * 1.5:

    st.warning(
        "High consumption expected in the next 15 minutes."
    )

    st.write(
        "Consider switching off unnecessary appliances "
        "such as lights, fans or high-power devices."
    )

else:

    st.success(
        "Energy consumption is currently within the normal range."
    )


# --------------------------------------------------
# 15. Energy-saving recommendation
# --------------------------------------------------

st.subheader("💡 Smart Energy-Saving Recommendation")

if next_prediction > average_consumption * 1.5:

    st.warning(
        "⚠️ High energy consumption is predicted in the next 15 minutes."
    )

    st.info(
        "💡 Consider switching off unnecessary high-power appliances "
        "such as air conditioners, heaters, washing machines or other "
        "heavy electrical devices."
    )

elif next_prediction > average_consumption:

    st.info(
        "📊 Energy consumption is expected to be slightly above the "
        "household average. Consider reducing unnecessary appliance usage."
    )

else:

    st.success(
        "✅ Energy consumption is predicted to remain within the normal range. "
        "Continue monitoring high-power appliances during peak periods."
    )


# --------------------------------------------------
# 16. Last updated
# --------------------------------------------------

st.caption(
    f"Latest available reading: {latest_time}"
)
st.divider()

st.subheader("📌 About This Project")

st.write(
    """
    This application uses a Long Short-Term Memory (LSTM) neural network
    to forecast household electricity consumption.

    The model uses the previous 24 hours of 15-minute electricity
    consumption readings to predict energy usage for the next hour.

    The dashboard provides:
    - Historical energy consumption analysis
    - Peak usage identification
    - Hourly consumption patterns
    - Next 15–60 minute energy forecasts
    - High-consumption alerts
    - Smart energy-saving recommendations
    """
)

st.caption(
    "Model: LSTM | Forecast Horizon: 15–60 minutes | "
    "Input Window: Previous 24 hours"
)