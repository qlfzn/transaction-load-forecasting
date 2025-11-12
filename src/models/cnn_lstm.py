import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def create_sequences(data, lookback, multivariate=False):
    """
    Create sequences for time series prediction
    
    Args:
        data: array of time series values (univariate) or DataFrame (multivariate)
        lookback: number of previous time steps to use as input
        multivariate: whether to use multivariate mode
    
    Returns:
        X: input sequences (samples, lookback, features)
        y: target values (samples,)
    """
    if multivariate:
        # data should be a DataFrame with features
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data.iloc[i:i+lookback].values)  # (lookback, n_features)
            y.append(data.iloc[i+lookback]['txn_count'])  # target only
        return np.array(X), np.array(y)
    else:
        # Original univariate logic
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

def run_cnn_lstm(train_data, test_data, lookback=10, epochs=30, batch_size=64,
                 use_early_stopping=True, verbose=1, multivariate=False):
    """
    Train and evaluate a CNN-LSTM model for TPS forecasting
    
    Args:
        train_data: training data (Series for univariate, DataFrame for multivariate)
        test_data: test data (Series for univariate, DataFrame for multivariate)
        lookback: number of previous time steps to use (default: 10)
        epochs: number of training epochs (default: 30)
        batch_size: batch size for training (default: 64)
        use_early_stopping: whether to use early stopping (default: True)
        verbose: training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        multivariate: whether to use multivariate mode (default: False)
    
    Returns:
        predictions: predicted values for test set
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    
    mode = "Multivariate" if multivariate else "Univariate"
    print("\n  ⚙️  CNN-LSTM Configuration ({})".format(mode))
    print(f"     • Lookback window: {lookback} minutes")
    print(f"     • Batch size: {batch_size}")
    print(f"     • Max epochs: {epochs}")
    
    # Prepare sequences
    print("\n   Preparing sequences...")
    if multivariate:
        X_train, y_train = create_sequences(train_data, lookback, multivariate=True)
        X_test, y_test = create_sequences(test_data, lookback, multivariate=True)
        n_features = X_train.shape[2]
    else:
        X_train, y_train = create_sequences(train_data.values, lookback, multivariate=False)
        X_test, y_test = create_sequences(test_data.values, lookback, multivariate=False)
        # Reshape for CNN-LSTM: (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        n_features = 1
    
    print(f"     • Training sequences: {X_train.shape[0]:,}")
    print(f"     • Test sequences: {X_test.shape[0]:,}")
    print(f"     • Input shape: ({X_train.shape[1]}, {n_features}) (timesteps, features)")
    
    # Build CNN-LSTM model
    print("\n  ⚙️  Building CNN-LSTM model...")
    model = Sequential([
        Conv1D(
            64,
            kernel_size=3,
            activation='relu',
            input_shape=(lookback, n_features),
            name='conv1d_layer'
        ),
        MaxPooling1D(pool_size=2, name='maxpool_layer'),
        
        LSTM(
            64,
            activation='tanh',
            name='lstm_layer'
        ),
        Dropout(0.2),
        
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
    print("\n   Training CNN-LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=0.1
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
    
    # Save model
    os.makedirs('src/models/saved', exist_ok=True)
    model.save('src/models/saved/cnn_lstm_model.keras')
    print("  ✓ Model saved to models/saved/cnn_lstm_model.keras")
    
    return preds, mae, rmse
