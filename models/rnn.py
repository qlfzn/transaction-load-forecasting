import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def create_sequences(data, lookback):
    """
    Create sequences for time series prediction
    
    Args:
        data: array of time series values
        lookback: number of previous time steps to use as input
    
    Returns:
        X: input sequences (samples, lookback)
        y: target values (samples,)
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def run_rnn(train, test, lookback=10, epochs=30, batch_size=64, units=64, 
            dropout=0.2, use_early_stopping=True, verbose=1):
    """
    Train and evaluate a vanilla RNN model for TPS forecasting
    
    Args:
        train: training time series (pandas Series)
        test: test time series (pandas Series)
        lookback: number of previous time steps to use (default: 10)
        epochs: number of training epochs (default: 30)
        batch_size: batch size for training (default: 32)
        units: number of RNN units in each layer (default: 64)
        dropout: dropout rate for regularization (default: 0.2)
        use_early_stopping: whether to use early stopping (default: True)
        verbose: training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        predictions: predicted values for test set
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    
    print("\n  ️  RNN Configuration:")
    print(f"     • Lookback window: {lookback} minutes")
    print(f"     • RNN units: {units}")
    print(f"     • Dropout rate: {dropout}")
    print(f"     • Batch size: {batch_size}")
    print(f"     • Max epochs: {epochs}")
    
    # Prepare sequences
    print("\n   Preparing sequences...")
    X_train, y_train = create_sequences(train.values, lookback)
    X_test, y_test = create_sequences(test.values, lookback)
    
    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"     • Training sequences: {X_train.shape[0]:,}")
    print(f"     • Test sequences: {X_test.shape[0]:,}")
    print(f"     • Input shape: {X_train.shape[1:]} (timesteps, features)")
    
    # Build RNN model
    print("\n  ️  Building RNN model...")
    model = Sequential([
        # First RNN layer with return_sequences=True (stacks layers)
        SimpleRNN(
            units=units, 
            activation='tanh',  # RNN typically uses tanh
            return_sequences=True,
            input_shape=(lookback, 1),
            name='rnn_layer_1'
        ),
        Dropout(dropout),
        
        # Second RNN layer (no return_sequences for final output)
        SimpleRNN(
            units=units//2,  # Reduce units in second layer
            activation='tanh',
            name='rnn_layer_2'
        ),
        Dropout(dropout),
        
        # Dense output layer
        Dense(1, name='output_layer')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    print("     • Model architecture:")
    model.summary(print_fn=lambda x: print(f"       {x}"))
    
    # Setup callbacks
    callbacks = []
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stop)
        print("\n     • Early stopping enabled (patience: 5)")
    
    # Train model
    print("\n   Training RNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=0.1  # Use 10% of training data for validation
    )
    
    # Get actual epochs trained
    actual_epochs = len(history.history['loss'])
    print(f"\n     ✓ Training completed in {actual_epochs} epochs")
    
    # Make predictions
    print("\n   Making predictions...")
    preds = model.predict(X_test, verbose=0)
    preds = preds.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Additional metrics
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    # Compare with baseline (naive forecast: last value)
    baseline_preds = X_test[:, -1, 0]  # Use last value as prediction
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    
    # Save model (optional)
    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/rnn_model.keras')
    print("  ✓ Model saved to models/saved/rnn_model.keras")
    
    return preds, mae, rmse