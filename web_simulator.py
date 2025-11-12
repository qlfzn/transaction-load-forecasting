"""
Transaction Load Forecasting - Streamlit Dashboard
Simple, beautiful, and interactive visualization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model # type: ignore
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Transaction Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_loaded = False
    st.session_state.current_index = 0
    st.session_state.predictions = []
    st.session_state.actuals = []
    st.session_state.timestamps = []
    st.session_state.recent_window = []
    st.session_state.is_running = False

# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_resource
def load_data():
    """Load and prepare data"""
    df = pd.read_csv('data/synthetic_fraud_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    df_tps = df.set_index('timestamp').resample('1min').size().reset_index(name='tps')
    df_tps['tps'] = df_tps['tps'].fillna(0)
    return df_tps

@st.cache_resource
def load_prediction_model(model_path):
    """Load trained model"""
    try:
        model = load_model(model_path)
        return model, True
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}. Using baseline prediction.")
        return None, False

def predict_next(model, recent_values, lookback=10):
    """Make prediction"""
    if model is not None and len(recent_values) >= lookback:
        X = np.array(recent_values[-lookback:]).reshape(1, lookback, 1)
        pred = model.predict(X, verbose=0)
        return float(pred[0][0])
    else:
        # Baseline: 3-minute moving average
        return np.mean(recent_values[-3:]) if len(recent_values) >= 3 else recent_values[-1]

def calculate_metrics(predictions, actuals):
    """Calculate performance metrics"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    return mae, rmse, mape

# ============================================
# HEADER
# ============================================
st.markdown('<div class="main-header"> Transaction Load Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time prediction visualization</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - CONTROLS
# ============================================
st.sidebar.header("⚙️ Controls")

# Model selection
model_options = {
    'RNN': 'models/saved/rnn_model.keras',
    'LSTM': 'models/saved/lstm_model.keras',
    'CNN-LSTM': 'models/saved/cnn_lstm_model.keras',
    'Baseline (Moving Avg)': None
}
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0
)

# Configuration
lookback = st.sidebar.slider("Lookback Window", 5, 30, 10, 
                             help="Number of previous minutes to use for prediction")
test_size = st.sidebar.slider("Test Size (minutes)", 50, 1000, 300, 50,
                              help="Number of minutes to visualize")
window_size = st.sidebar.slider("Chart Window", 20, 100, 50, 10,
                                help="Number of points to display on chart")
speed = st.sidebar.slider("Speed (updates/sec)", 1, 20, 5, 1,
                          help="How fast to process predictions")

st.sidebar.markdown("---")

# Load data button
if st.sidebar.button("🔄 Load Data", use_container_width=True):
    with st.spinner("Loading data..."):
        df_tps = load_data()
        st.session_state.df_tps = df_tps
        st.session_state.test_data = df_tps.tail(test_size).reset_index(drop=True)
        st.session_state.data_loaded = True
        st.session_state.current_index = lookback
        st.session_state.recent_window = st.session_state.test_data['tps'].head(lookback).tolist()
        st.session_state.predictions = []
        st.session_state.actuals = []
        st.session_state.timestamps = []
        st.success(f"✓ Loaded {len(df_tps):,} minutes of data")

# Load model button
if st.sidebar.button("🧠 Load Model", use_container_width=True, disabled=not st.session_state.data_loaded):
    if model_options[selected_model] is not None:
        with st.spinner(f"Loading {selected_model} model..."):
            model, success = load_prediction_model(model_options[selected_model])
            st.session_state.model = model
            st.session_state.model_loaded = success
            if success:
                st.success(f"✓ {selected_model} model loaded")
    else:
        st.session_state.model = None
        st.session_state.model_loaded = True
        st.info("Using baseline moving average prediction")

st.sidebar.markdown("---")

# Control buttons
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("▶️ Start", use_container_width=True, 
                           disabled=not st.session_state.data_loaded or st.session_state.is_running)
stop_button = col2.button("⏸️ Stop", use_container_width=True,
                         disabled=not st.session_state.is_running)

if start_button:
    st.session_state.is_running = True
    
if stop_button:
    st.session_state.is_running = False

if st.sidebar.button("🔄 Reset", use_container_width=True, disabled=not st.session_state.data_loaded):
    st.session_state.current_index = lookback
    st.session_state.predictions = []
    st.session_state.actuals = []
    st.session_state.timestamps = []
    st.session_state.recent_window = st.session_state.test_data['tps'].head(lookback).tolist()
    st.session_state.is_running = False
    st.rerun()

# ============================================
# MAIN CONTENT
# ============================================

if not st.session_state.data_loaded:
    # Welcome screen
    st.info("👈 Click 'Load Data' in the sidebar to begin")
    
    st.markdown("### 📚 Quick Start Guide")
    st.markdown("""
    1. **Load Data** - Click the button to load transaction data
    2. **Load Model** - Choose and load a prediction model
    3. **Start** - Begin the live prediction simulation
    4. **Adjust** - Use sliders to control speed and display
    
    #### Features:
    - 🎯 Real-time predictions with multiple model options
    - 📊 Live updating charts with smooth animations  
    - 📈 Performance metrics (MAE, RMSE, MAPE)
    - ⚡ Adjustable speed and window size
    - 🔄 Pause, resume, and reset controls
    """)
    
else:
    # Main dashboard
    
    # Top metrics row
    if len(st.session_state.predictions) > 0:
        mae, rmse, mape = calculate_metrics(
            st.session_state.predictions,
            st.session_state.actuals
        )
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Predicted",
                f"{st.session_state.predictions[-1]:.1f}",
                help="Latest predicted TPS"
            )
        
        with col2:
            st.metric(
                "Current Actual", 
                f"{st.session_state.actuals[-1]:.1f}",
                help="Latest actual TPS"
            )
        
        with col3:
            error = abs(st.session_state.predictions[-1] - st.session_state.actuals[-1])
            st.metric(
                "Current Error",
                f"{error:.1f}",
                help="Absolute difference"
            )
        
        with col4:
            st.metric(
                "MAE",
                f"{mae:.2f}",
                delta=f"{mape:.1f}%",
                delta_color="inverse",
                help="Mean Absolute Error & MAPE"
            )
        
        with col5:
            progress = (st.session_state.current_index / len(st.session_state.test_data)) * 100
            st.metric(
                "Progress",
                f"{progress:.1f}%",
                f"{len(st.session_state.predictions)} min",
                help="Predictions completed"
            )
    else:
        st.info("Click 'Start' to begin predictions")
    
    st.markdown("---")
    
    # Create placeholder for charts
    chart_placeholder = st.empty()
    
    # Create charts
    if len(st.session_state.predictions) > 0:
        # Get last N points for display
        display_size = min(window_size, len(st.session_state.predictions))
        x_data = list(range(len(st.session_state.predictions) - display_size, len(st.session_state.predictions)))
        actual_data = st.session_state.actuals[-display_size:]
        predicted_data = st.session_state.predictions[-display_size:]
        error_data = [abs(p - a) for p, a in zip(predicted_data, actual_data)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Predicted vs Actual TPS', 'Prediction Error'),
            vertical_spacing=0.12
        )
        
        # Main line chart
        fig.add_trace(
            go.Scatter(
                x=x_data, y=actual_data,
                mode='lines+markers',
                name='Actual TPS',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=6),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_data, y=predicted_data,
                mode='lines+markers',
                name='Predicted TPS',
                line=dict(color='#FF5722', width=3, dash='dash'),
                marker=dict(size=6, symbol='square'),
            ),
            row=1, col=1
        )
        
        # Error bar chart
        fig.add_trace(
            go.Bar(
                x=x_data, y=error_data,
                name='Error',
                marker=dict(color='#FF9800', opacity=0.6),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        
        fig.update_xaxes(title_text="Time Index", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=1)
        fig.update_yaxes(title_text="Transactions per Minute", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Absolute Error", row=2, col=1, showgrid=True, gridcolor='lightgray')
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Auto-run simulation
    if st.session_state.is_running:
        if st.session_state.current_index < len(st.session_state.test_data):
            # Get actual value
            actual_tps = st.session_state.test_data['tps'].iloc[st.session_state.current_index]
            
            # Make prediction
            predicted_tps = predict_next(
                st.session_state.model if st.session_state.model_loaded else None,
                st.session_state.recent_window,
                lookback
            )
            
            # Store results
            st.session_state.predictions.append(predicted_tps)
            st.session_state.actuals.append(actual_tps)
            st.session_state.timestamps.append(st.session_state.current_index)
            
            # Update window
            st.session_state.recent_window.append(actual_tps)
            if len(st.session_state.recent_window) > lookback:
                st.session_state.recent_window.pop(0)
            
            # Move to next
            st.session_state.current_index += 1
            
            # Control speed
            time.sleep(1.0 / speed)
            
            # Rerun to update
            st.rerun()
        else:
            st.session_state.is_running = False
            st.success("✓ Completed all predictions!")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 About
Transaction load forecasting system using deep learning.

**Tech Stack:**
- Python & TensorFlow
- Streamlit
- Plotly

**Models:**
- RNN, LSTM, CNN-LSTM
- Statistical baselines

[GitHub](#) | [Docs](#)
""")