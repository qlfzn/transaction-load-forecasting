"""
Configuration details for pipeline
"""
CONFIG = {
    'data_path': 'data/synthetic_fraud_data.csv',
    'train_ratio': 0.8,
    'resample_freq': '1min',
    'lstm_lookback': 96,
    'lstm_epochs': 50,
    'cnn_lstm_lookback': 96,
    'cnn_lstm_epochs': 50,
    'rnn_lookback': 96,
    'rnn_epochs': 50
}