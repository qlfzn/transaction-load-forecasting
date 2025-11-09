import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/synthetic_fraud_data.csv',
    'model_path': 'models/saved/rnn_model.keras',
    'lookback': 10,
    'test_size': 0.1,
    'display_interval': 1.0,
}

# Load & prepare data
print("=" * 70)
print("SIMPLE TRANSACTION PREDICTION SIMULATOR")
print("=" * 70)
print("\n[1/4] Loading transaction data...")

df = pd.read_csv(CONFIG['data_path'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")

if df['timestamp'].dt.tz is not None:
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

print(f"  ✓ Loaded {len(df):,} transactions")
print(f"  ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# Aggregate to per-minute TPS
print("\n[2/4] Aggregating to per-minute TPS...")
df_agg = df.set_index('timestamp').resample('1min').size().reset_index(name='tps')
df_agg['tps'] = df_agg['tps'].fillna(0)

print(f"  ✓ Aggregated to {len(df_agg):,} minutes")
print(f"  ✓ Average TPS: {df_agg['tps'].mean():.2f} txn/min")
print(f"  ✓ Max TPS: {df_agg['tps'].max():.0f} txn/min")
print(f"  ✓ Min TPS: {df_agg['tps'].min():.0f} txn/min")

# Split into train-test
split_idx = int(len(df_agg) * (1 - CONFIG['test_size']))
train_data = df_agg[:split_idx].copy()
test_data = df_agg[split_idx:].copy()

print("\n  Dataset Split:")
print(f"  ├─ Training: {len(train_data):,} minutes ({len(train_data)/len(df_agg)*100:.1f}%)")
print(f"  └─ Testing:  {len(test_data):,} minutes ({len(test_data)/len(df_agg)*100:.1f}%)")

# Load model
print("\n[3/4] Loading prediction model...")
try:
    model = load_model(CONFIG['model_path'])
    print(f"  ✓ Model loaded from {CONFIG['model_path']}")
    model_loaded = True
except Exception as e:
    print(f"  ✗ Could not load model: {e}")
    print("  → Will show baseline prediction (moving average)")
    model_loaded = False

def predict_next_minute(model, recent_tps, lookback=10):
    """Make prediction using loaded model"""
    if len(recent_tps) < lookback:
        return None
    
    X = np.array(recent_tps[-lookback:]).reshape(1, lookback, 1)
    pred = model.predict(X, verbose=0)
    return float(pred[0][0])

def baseline_predict(recent_tps):
    """Simple baseline: moving average of last 3 minutes"""
    if len(recent_tps) < 3:
        return recent_tps[-1] if recent_tps else 0
    return np.mean(recent_tps[-3:])

def calculate_metrics(predictions, actuals):
    """Calculate prediction accuracy metrics"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    return mae, rmse, mape

def print_live_prediction(minute_idx, timestamp, actual_tps, predicted_tps, 
                         recent_tps, running_metrics=None):
    """Print live prediction dashboard"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 70)
    print(" TRANSACTION PREDICTION SIMULATOR")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Progress:  {minute_idx}/{len(test_data)} minutes")
    print("-" * 70)
    
    # Historical context
    print(f"\n Recent TPS (last {len(recent_tps)} minutes):")
    if len(recent_tps) >= 5:
        recent_5 = recent_tps[-5:]
        print(f"  {' → '.join([f'{int(x):3d}' for x in recent_5])}")
    else:
        print(f"  {recent_tps}")
    
    # Prediction
    error = abs(predicted_tps - actual_tps)
    error_pct = (error / actual_tps * 100) if actual_tps > 0 else 0
    
    print("\n Current Minute Prediction:")
    print(f"  ├─ Predicted TPS:  {predicted_tps:>6.1f} txn/min")
    print(f"  ├─ Actual TPS:     {actual_tps:>6.0f} txn/min")
    print(f"  └─ Error:          {error:>6.1f} ({error_pct:.1f}%)")
    
    # Running metrics
    if running_metrics:
        mae, rmse, mape = running_metrics
        print("\n Running Performance (so far):")
        print(f"  ├─ MAE:   {mae:>6.2f} transactions")
        print(f"  ├─ RMSE:  {rmse:>6.2f} transactions")
        print(f"  └─ MAPE:  {mape:>6.2f}%")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop...")



print("\n[4/4] Starting predictions on test data...")
print(f"  → Will predict {len(test_data)} minutes")
print(f"  → Using last {CONFIG['lookback']} minutes as input")
print("\n  Starting in 3 seconds...\n")
time.sleep(3)

predictions = []
actuals = []

lookback = CONFIG['lookback']
recent_tps = train_data['tps'].tail(lookback).tolist()

try:
    for idx, row in test_data.iterrows():
        actual_tps = row['tps']
        timestamp = row['timestamp']
        
        # Make prediction
        if model_loaded:
            predicted_tps = predict_next_minute(model, recent_tps, lookback)
        else:
            predicted_tps = baseline_predict(recent_tps)
        
        if predicted_tps is None:
            predicted_tps = actual_tps  # Fallback
        
        # Store results
        predictions.append(predicted_tps)
        actuals.append(actual_tps)
        
        # Calculate running metrics
        if len(predictions) > 1:
            running_metrics = calculate_metrics(predictions, actuals)
        else:
            running_metrics = None
        
        # Display
        minute_idx = len(predictions)
        print_live_prediction(
            minute_idx, timestamp, actual_tps, predicted_tps, 
            recent_tps[-10:], running_metrics
        )
        
        # Update recent_tps window (slide forward)
        recent_tps.append(actual_tps)
        if len(recent_tps) > lookback:
            recent_tps.pop(0)
        
        # Pause for visibility (remove this for instant processing)
        time.sleep(CONFIG['display_interval'])

except KeyboardInterrupt:
    print("\n\n⏸  Simulation stopped by user")


print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

if len(predictions) > 0:
    mae, rmse, mape = calculate_metrics(predictions, actuals)
    
    print(f"\nPredictions completed: {len(predictions)} minutes")
    print("\n Overall Performance:")
    print(f"  ├─ MAE (Mean Absolute Error):       {mae:.2f} transactions")
    print(f"  ├─ RMSE (Root Mean Squared Error):  {rmse:.2f} transactions")
    print(f"  └─ MAPE (Mean Absolute % Error):    {mape:.2f}%")
    
    avg_actual = np.mean(actuals)
    print("\n Context:")
    print(f"  ├─ Average actual TPS:  {avg_actual:.2f} txn/min")
    print(f"  ├─ Prediction accuracy: {100 - mape:.2f}%")
    print(f"  └─ Model type:          {'AI Model' if model_loaded else 'Baseline (Moving Avg)'}")
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': test_data['timestamp'].values,
        'actual_tps': actuals,
        'predicted_tps': predictions,
        'error': np.abs(np.array(predictions) - np.array(actuals))
    })
    
    output_path = 'predictions_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n Results saved to: {output_path}")

else:
    print("No predictions were made.")

print("=" * 70)