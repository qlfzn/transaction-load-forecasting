import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_predictions(actuals, predictions, model_name, save_dir='results/plots', lookback=0):
    """
    Create actual vs predicted line plot for a single model
    
    Args:
        actuals: Array, Series, or DataFrame of actual values (DataFrame should have 'txn_count' column in multivariate mode)
        predictions: Array or Series of predicted values
        model_name: Name of the model (for title and filename)
        save_dir: Directory to save plots
        lookback: Number of lookback steps to skip (default: 0)
    
    Returns:
        None (saves plot to file)
    """
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if needed
    # Handle DataFrame case (multivariate mode)
    if hasattr(actuals, 'columns'):
        # It's a DataFrame - extract txn_count column
        if 'txn_count' in actuals.columns:
            actuals = actuals['txn_count'].values
        else:
            # Fallback: use first column
            actuals = actuals.iloc[:, 0].values
    elif hasattr(actuals, 'values'):
        # It's a Series
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