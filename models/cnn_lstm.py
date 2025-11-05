import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def run_cnn_lstm(train, test, lookback=10, epochs=10, use_early_stopping=True):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(train).reshape(-1,1))
    X, y = create_sequences(scaled, lookback)
    
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(lookback, 1)),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary(print_fn=lambda x: print(f"       {x}"))

    # Setup callbacks
    callbacks = []
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='loss',
            patience=7,
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stop)
        print("\n     • Early stopping enabled (patience: 5)")

    model.fit(X, y, epochs=epochs, batch_size=64, verbose=1)
    
    # Forecast
    input_seq = scaled[-lookback:]
    preds = []
    for _ in range(len(test)):
        pred = model.predict(input_seq.reshape(1, lookback, 1), verbose=0)
        preds.append(pred[0,0])
        input_seq = np.append(input_seq[1:], pred)[-lookback:]

    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/cnn_lstm_model.keras')
    print("  ✓ Model saved to models/saved/cnn_lstm_model.keras")
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    
    return preds, mae, rmse
