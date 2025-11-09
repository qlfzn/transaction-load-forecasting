# experiment_runner.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import mlflow
from mlflow import tensorflow

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
EPOCHS = 100
BASE_LOOKBACKS = [10, 30, 60]
BATCH_SIZES = [32, 64]
MODELS = ['RNN', 'LSTM', 'GRU']

# ---------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------
def load_data(path="data/txn_counts.csv"):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    df = df.set_index('timestamp').resample("1min").size().reset_index(name='txn_count')
    df['txn_count'] = df['txn_count'].fillna(0)
    df['txn_count'] = df['txn_count'].astype(float)
    print(f"\nFinished reading data: {df.count()} records.")
    return df

def add_time_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['txn_ma_5'] = df['txn_count'].rolling(5, min_periods=1).mean()
    return df.drop(columns=['hour'])

# ---------------------------------------------------
# Create sequences
# ---------------------------------------------------
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])  # first column = txn_count
    return np.array(X), np.array(y)

# ---------------------------------------------------
# Build model
# ---------------------------------------------------
def build_model(model_type, input_shape, units=64, dropout=0.2):
    model = Sequential()
    if model_type == "RNN":
        model.add(SimpleRNN(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(units//2))
    elif model_type == "LSTM":
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units//2))
    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(GRU(units//2))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------------------------------
# Train and evaluate
# ---------------------------------------------------
def train_and_evaluate(model_type, lookback, batch_size, df):
    df_feat = add_time_features(df)
    print(df_feat.head(10))
    num_cols = df_feat.select_dtypes(include=['number']).columns
    values = df[num_cols]

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Split
    split_idx = int(len(scaled) * 0.8)
    train, test = scaled[:split_idx], scaled[split_idx:]

    # Sequences
    X_train, y_train = create_sequences(train, lookback)
    X_test, y_test = create_sequences(test, lookback)

    # Build model
    model = build_model(model_type, (X_train.shape[1], X_train.shape[2]))

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=0)

    # Fit
    hist = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Predict
    preds = model.predict(X_test, verbose=0)
    preds_unscaled = scaler.inverse_transform(
        np.concatenate([preds, np.zeros((len(preds), scaled.shape[1]-1))], axis=1)
    )[:,0]
    y_test_unscaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), scaled.shape[1]-1))], axis=1)
    )[:,0]

    # Metrics
    mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, preds_unscaled))
    mape = mean_absolute_percentage_error(y_test_unscaled, preds_unscaled) * 100

    return model, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ---------------------------------------------------
# Run experiments and log to MLflow
# ---------------------------------------------------
def run_experiments(data_path="data/txn_counts.csv"):
    df = load_data(data_path)

    for model_type in MODELS:
        for lookback in BASE_LOOKBACKS:
            for batch in BATCH_SIZES:
                with mlflow.start_run(run_name=f"{model_type}_lb{lookback}_b{batch}"):
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("lookback", lookback)
                    mlflow.log_param("batch_size", batch)

                    model, metrics = train_and_evaluate(model_type, lookback, batch, df)

                    mlflow.log_metrics(metrics)
                    tensorflow.log_model(model, f"{model_type}_model")

                    print(f"✅ {model_type} (lb={lookback}, b={batch}) → "
                          f"MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

if __name__ == "__main__":
    mlflow.set_experiment("Transaction_Load_Improved")
    run_experiments("data/synthetic_fraud_data.csv")
