import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def run_lstm(train, test, lookback=10, epochs=30, use_early_stopping=5):
    # Prepare sequences
    X_train, y_train = create_sequences(train.values, lookback)
    X_test, y_test = create_sequences(test.values, lookback)
    
    print(f"Training shape: {X_train.shape}")  # Should see this!
    
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(lookback, 1)),
        Dense(1),
        Dropout(0.3)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary(print_fn=lambda x: print(f"       {x}"))

    # Setup callbacks
    callbacks = []
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='loss',
            patience=6,
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stop)
        print("\n     • Early stopping enabled (patience: 5)")
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
    
    # SAVE THE MODEL
    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/lstm_model.keras')
    print("  ✓ Model saved to models/saved/lstm_model.keras")
    
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds.flatten() - y_test))
    rmse = np.sqrt(np.mean((preds.flatten() - y_test)**2))
    
    return preds, mae, rmse