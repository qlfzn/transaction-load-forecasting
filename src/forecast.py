import os
import warnings
import pandas as pd
import mlflow
from datetime import datetime

from utils import loader, preprocessor, visualiser
from models.lstm import run_lstm
from models.cnn_lstm import run_cnn_lstm
from models.rnn import run_rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

CONFIG = {
    'data_path': 'data/synthetic_fraud_data.csv',
    'multivariate': True,
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
df = loader.load_data(CONFIG['data_path'])
print(f"  ✓ Loaded {len(df):,} transaction records")

# Convert to time series
df = preprocessor.set_timeseries(df, resample_freq=CONFIG['resample_freq'])

print(f"  ✓ Resampled to {len(df):,} time steps ({CONFIG['resample_freq']})")
print(f"  ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  ✓ Time taken: {(datetime.now() - start_time).seconds}s")

# Add time-series features
df = preprocessor.add_time_features(df)

# Set index and split - ALWAYS return full DataFrames with features + target
train_df, test_df = preprocessor.prepare_train_data(
    df=df, 
    train_frac=CONFIG['train_ratio'],
    multivariate=CONFIG['multivariate']
)

print("\n  Dataset Split:")
if CONFIG['multivariate']:
    print(f"  - Features: {list(train_df.columns)}")
    print(f"  - Training: {len(train_df):,} steps ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Testing:  {len(test_df):,} steps ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  - Avg TXN_COUNT (train): {train_df['txn_count'].mean():.2f}")
    print(f"  - Avg TXN_COUNT (test):  {test_df['txn_count'].mean():.2f}")
else:
    print("  - Mode: Univariate (txn_count only)")
    print(f"  - Training: {len(train_df):,} steps ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Testing:  {len(test_df):,} steps ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  - Avg TPS (train): {train_df.mean():.2f}")
    print(f"  - Avg TPS (test):  {test_df.mean():.2f}")

# Setup Mlflow
print("\n[2/6] Setting up MLflow...")
mlflow.set_experiment("Transaction_Load_Forecasting")
print("  ✓ Experiment: Transaction_Load_Forecasting")

batch_run_name = f"Batch_{datetime.now().strftime('%Y%m%d_%H:%M:%S')}"

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
                # Simply pass train_df/test_df - function handles both modes
                preds, mae, rmse = run_function(
                    train_df, test_df,
                    lookback=params.get('lookback'),
                    epochs=params.get('epochs'),
                    multivariate=CONFIG['multivariate']
                )
                
                # Get avg TPS appropriately
                if CONFIG['multivariate']:
                    avg_tps = float(test_df['txn_count'].mean())  # type: ignore
                else:
                    # test_df is a Series in univariate mode
                    avg_tps = float(test_df.mean())  # type: ignore
                
                # Calculate additional metrics
                mape = (mae / avg_tps) * 100 if avg_tps > 0.0 else 0.0
                
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
                import traceback
                traceback.print_exc()
                return False, None, None

print("\n[4/6] Training LSTM...")
success, preds_lstm, mae_lstm = train_and_log_model(
    model_name="LSTM",
    run_function=run_lstm,
    params={
        "model_type": "LSTM",
        "lookback": CONFIG['lstm_lookback'],
        "epochs": CONFIG['lstm_epochs'],
        "units": 64
    },
    run_name="LSTM_Model",
    parent_run_id=parent_run.info.run_id
)
if success:
    print("\n   Creating prediction plot for LSTM...")
    visualiser.plot_predictions(test_df, preds_lstm, "LSTM", lookback=CONFIG['lstm_lookback'])

print("\n[5/6] Training CNN-LSTM...")
success, preds_cnn_lstm, mae_cnn_lstm = train_and_log_model(
    model_name="CNN-LSTM",
    run_function=run_cnn_lstm,
    params={
        "model_type": "CNN-LSTM",
        "lookback": CONFIG['cnn_lstm_lookback'],
        "epochs": CONFIG['cnn_lstm_epochs'],
        "filters": 64
    },
    run_name="CNN_LSTM_Model",
    parent_run_id=parent_run.info.run_id
)

if success:
    print("\n   Creating prediction plot for CNN_LSTM...")
    visualiser.plot_predictions(test_df, preds_cnn_lstm, "CNN_LSTM", lookback=CONFIG['cnn_lstm_lookback'])

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
    visualiser.plot_predictions(test_df, preds_rnn, "RNN", lookback=CONFIG['rnn_lookback'])

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n Best Model:")
    best_model = results_df.loc[results_df['MAE'].idxmin()]
    print(f"  {best_model['Model']} with MAE: {best_model['MAE']:.3f}")
else:
    print("No models completed successfully.")

print("\n✓ Pipeline completed!")
print("✓ View results: mlflow ui")
print("=" * 60)