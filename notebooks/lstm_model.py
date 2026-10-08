import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "energy_15min.csv"
model_dir = BASE_DIR / "model"

model_dir.mkdir(exist_ok=True)

model_path = model_dir / "energy_lstm.keras"


# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

df = pd.read_csv(
    file_path,
    parse_dates=["datetime"],
    index_col="datetime"
)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Select target
# --------------------------------------------------

data = df[["Global_active_power"]].values

print("Data shape:", data.shape)


# --------------------------------------------------
# 4. Train/Test split
# --------------------------------------------------

train_size = int(len(data) * 0.80)

train_data = data[:train_size]
test_data = data[train_size:]

print("\nTraining data:", train_data.shape)
print("Testing data:", test_data.shape)


# --------------------------------------------------
# 5. Scale data
# --------------------------------------------------

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


# --------------------------------------------------
# 6. Create sequences
# --------------------------------------------------

sequence_length = 96


def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(sequence_length, len(data)):

        X.append(
            data[i - sequence_length:i]
        )

        y.append(
            data[i]
        )

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_scaled,
    sequence_length
)

X_test, y_test = create_sequences(
    test_scaled,
    sequence_length
)


print("\nSequence shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 7. Build LSTM model
# --------------------------------------------------

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(sequence_length, 1)
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# --------------------------------------------------
# 8. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nModel summary:")
model.summary()


# --------------------------------------------------
# 9. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

model.save(model_path)

print("\nModel saved successfully!")
print("Model path:", model_path)


# --------------------------------------------------
# 12. Predict test data
# --------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=1
)


# --------------------------------------------------
# 13. Convert predictions back to kW
# --------------------------------------------------

y_test_actual = scaler.inverse_transform(y_test)

predictions_actual = scaler.inverse_transform(predictions)


# --------------------------------------------------
# 14. Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test_actual,
    predictions_actual
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        predictions_actual
    )
)


print("\nModel Evaluation")
print("-----------------------")

print("MAE :", mae, "kW")
print("RMSE:", rmse, "kW")


# --------------------------------------------------
# 15. Plot Actual vs Predicted
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(
    y_test_actual[:500],
    label="Actual"
)

plt.plot(
    predictions_actual[:500],
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Household Energy Consumption"
)

plt.xlabel("15-Minute Time Steps")

plt.ylabel("Global Active Power (kW)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# --------------------------------------------------
# 16. Plot training history
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()