import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from models.lstm import run_lstm
from models.cnn_lstm import run_cnn_lstm
from models.rnn import run_rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/synthetic_fraud_data.csv',
    'train_ratio': 0.8,
    'resample_freq': '1min',  # Can change to '5min', '15min', '1H' for faster testing
    'lstm_lookback': 60,
    'lstm_epochs': 50,
    'cnn_lstm_lookback': 60,
    'cnn_lstm_epochs': 50,
    'rnn_lookback': 60,
    'rnn_epochs': 50
}

print("=" * 60)
print("TRANSACTION LOAD FORECASTING PIPELINE")
print("=" * 60)

# Load & prepare data
print("\n[1/6] Loading data...")
start_time = datetime.now()

df = pd.read_csv(CONFIG['data_path'])
print(f"  ✓ Loaded {len(df):,} transaction records")

# Convert to time series
df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
df = df.set_index('timestamp').resample(CONFIG['resample_freq']).size().reset_index(name='txn_count')
df['txn_count'] = df['txn_count'].fillna(0)

print(f"  ✓ Resampled to {len(df):,} time steps ({CONFIG['resample_freq']})")
print(f"  ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  ✓ Time taken: {(datetime.now() - start_time).seconds}s")

# Set index and split
df = df.set_index('timestamp')
train_size = int(len(df) * CONFIG['train_ratio'])
train, test = df['txn_count'][:train_size], df['txn_count'][train_size:]

print("\n  Dataset Split:")
print(f"  - Training: {len(train):,} steps ({len(train)/len(df)*100:.1f}%)")
print(f"  - Testing:  {len(test):,} steps ({len(test)/len(df)*100:.1f}%)")
print(f"  - Avg TPS (train): {train.mean():.2f}")
print(f"  - Avg TPS (test):  {test.mean():.2f}")

# Setup Mlflow
print("\n[2/6] Setting up MLflow...")
mlflow.set_experiment("Transaction_Load_Forecasting")
print("  ✓ Experiment: Transaction_Load_Forecasting")

batch_run_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=batch_run_name) as parent_run:
    print(f"Started MLflow batch run: {batch_run_name}")

    # Model training & evaluation
    results = []

    def train_and_log_model(model_name, run_function, params, run_name, parent_run_id):
        """Helper function to train model and log to MLflow"""
        print(f"\n[{model_name}] Starting training...")
        model_start = datetime.now()
        
        with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id):
            try:
                # Run model
                if model_name == "ARIMA":
                    forecast, mae, rmse = run_function(train, test)
                else:
                    preds, mae, rmse = run_function(train, test, 
                                                    lookback=params.get('lookback', 10),
                                                    epochs=params.get('epochs', 30))
                
                # Calculate additional metrics
                avg_tps = test.mean()
                mape = (mae / avg_tps) * 100 if avg_tps > 0 else 0
                
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics({
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "avg_test_TPS": avg_tps
                })
                
                # Log dataset
                mlflow.log_artifact(CONFIG['data_path'])
                
                # Calculate training time
                training_time = (datetime.now() - model_start).seconds
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Print results
                print(f"  ✓ {model_name} completed in {training_time}s")
                print(f"    - MAE:  {mae:.3f} transactions")
                print(f"    - RMSE: {rmse:.3f} transactions")
                print(f"    - MAPE: {mape:.2f}%")
                
                # Store results
                results.append({
                    'Model': model_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'Time (s)': training_time
                })
                
                return True, preds, mae
                
            except Exception as e:
                print(f"  ✗ {model_name} failed: {str(e)}")
                return False, None, None

def plot_predictions(actuals, predictions, model_name, save_dir='results/plots', lookback=0):
    """
    Create actual vs predicted line plot for a single model
    
    Args:
        actuals: Array or Series of actual values
        predictions: Array or Series of predicted values
        model_name: Name of the model (for title and filename)
        save_dir: Directory to save plots
    
    Returns:
        None (saves plot to file)
    """
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if needed
    if hasattr(actuals, 'values'):
        actuals = actuals.values
    if hasattr(predictions, 'values'):
        predictions = predictions.values

    if lookback > 0:
        actuals = actuals[lookback:]
    
    # Ensure same length
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[:min_len]
    predictions = predictions[:min_len]
    
    # Calculate metrics
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Prediction Results', fontsize=16, fontweight='bold')
    
    # ============================================
    # Subplot 1: Actual vs Predicted (Full)
    # ============================================
    x = np.arange(len(actuals))
    
    ax1.plot(x, actuals, label='Actual TPS', color='#2E86AB', linewidth=2, alpha=0.8)
    ax1.plot(x, predictions, label='Predicted TPS', color='#A23B72', 
             linewidth=2, linestyle='--', alpha=0.8)
    
    ax1.set_title('Actual vs Predicted TPS (Full Test Set)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step (minutes)', fontsize=11)
    ax1.set_ylabel('Transactions per Minute', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%'
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================
    # Subplot 2: Actual vs Predicted (Zoomed - First 100 points)
    # ============================================
    zoom_points = min(100, len(actuals))
    x_zoom = np.arange(zoom_points)
    
    ax2.plot(x_zoom, actuals[:zoom_points], label='Actual TPS', 
             color='#2E86AB', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    ax2.plot(x_zoom, predictions[:zoom_points], label='Predicted TPS', 
             color='#A23B72', linewidth=2.5, marker='s', markersize=4, 
             linestyle='--', alpha=0.8)
    
    ax2.set_title(f'Zoomed View (First {zoom_points} minutes)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step (minutes)', fontsize=11)
    ax2.set_ylabel('Transactions per Minute', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{model_name.lower().replace(" ", "_")}_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved plot: {filepath}")
    
    plt.close()


# LSTM MODEL
# print("\n[4/6] Training LSTM...")
# success, preds_lstm, mae_lstm = train_and_log_model(
#     model_name="LSTM",
#     run_function=run_lstm,
#     params={
#         "model_type": "LSTM",
#         "lookback": CONFIG['lstm_lookback'],
#         "epochs": CONFIG['lstm_epochs'],
#         "units": 64
#     },
#     run_name="LSTM_Model",
#     parent_run_id=parent_run.info.run_id
# )

# if success:
#     print("\n   Creating prediction plot for LSTM...")
#     plot_predictions(test, preds_lstm, "LSTM")

# CNN-LSTM MODEL
# print("\n[5/6] Training CNN-LSTM...")
# success, preds_cnn_lstm, mae_cnn_lstm = train_and_log_model(
#     model_name="CNN-LSTM",
#     run_function=run_cnn_lstm,
#     params={
#         "model_type": "CNN-LSTM",
#         "lookback": CONFIG['cnn_lstm_lookback'],
#         "epochs": CONFIG['cnn_lstm_epochs'],
#         "filters": 64
#     },
#     run_name="CNN_LSTM_Model",
#     parent_run_id=parent_run.info.run_id
# )

# if success:
#     print("\n   Creating prediction plot for CNN-LSTM...")
#     plot_predictions(test, preds_cnn_lstm, "CNN-LSTM")

print("\n[6/6] Training RNN...")
success, preds_rnn, mae_rnn = train_and_log_model(
    model_name="RNN",
    run_function=run_rnn,
    params={
        "model_type": "RNN",
        "lookback": CONFIG['rnn_lookback'],
        "epochs": CONFIG['rnn_epochs'],
        "units": 64,
        "architecture": "Vanilla RNN"
    },
    run_name="RNN_Model",
    parent_run_id=parent_run.info.run_id
)

if success:
    print("\n   Creating prediction plot for RNN...")
    plot_predictions(test, preds_rnn, "RNN", lookback=CONFIG['rnn_lookback'])

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n🏆 Best Model:")
    best_model = results_df.loc[results_df['MAE'].idxmin()]
    print(f"  {best_model['Model']} with MAE: {best_model['MAE']:.3f}")
else:
    print("No models completed successfully.")

print("\n✓ Pipeline completed!")
print("✓ View results: mlflow ui")
print("=" * 60)