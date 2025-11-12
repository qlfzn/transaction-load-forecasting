import os
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(actuals, predictions, model_name, save_dir='results/plots', lookback=0):
    """
    Create actual vs predicted line plot for a single model
    
    Args:
        actuals: Array, Series, or DataFrame of actual values (normalized)
        predictions: Array or Series of predicted values (denormalized)
        model_name: Name of the model (for title and filename)
        save_dir: Directory to save plots
        lookback: Number of lookback steps to skip (default: 0)
    
    Returns:
        None (saves plot to file)
    """
    from utils import preprocessor
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if needed
    # Handle DataFrame case (multivariate mode)
    if hasattr(actuals, 'columns'):
        actuals = actuals['txn_count'].values
    elif hasattr(actuals, 'values'):
        actuals = actuals.values
    
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    
    # Skip lookback samples to align with predictions
    if lookback > 0:
        actuals = actuals[lookback:]
    
    # Denormalize actuals (they come in normalized form)
    actuals = preprocessor.denormalize_predictions(actuals)  # type: ignore
    
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
    
    # Full test set plot
    ax1.plot(actuals, label='Actual TPS', color='#1f77b4', linewidth=2)
    ax1.plot(predictions, label='Predicted TPS', color='#d62728', linestyle='--', linewidth=2)
    ax1.set_title(f'{model_name} - Prediction Results\nActual vs Predicted TPS (Full Test Set)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step (minutes)', fontsize=11)
    ax1.set_ylabel('Transactions per Minute', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Zoomed view (first 100 minutes)
    zoom_len = min(100, len(actuals))
    ax2.plot(actuals[:zoom_len], label='Actual TPS', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    ax2.plot(predictions[:zoom_len], label='Predicted TPS', color='#d62728', linestyle='--', linewidth=2, marker='s', markersize=4)
    ax2.set_title(f'Zoomed View (First {zoom_len} minutes)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step (minutes)', fontsize=11)
    ax2.set_ylabel('Transactions per Minute', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.replace(' ', '_').lower()}_predictions.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to {filepath}")
    
    plt.close()