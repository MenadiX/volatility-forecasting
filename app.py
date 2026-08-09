import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")

# --- Model definition---
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

# --- Cached loading functions ---
@st.cache_data
def load_data():
    ftse = pd.read_csv('data/ftse_processed.csv', index_col='Date', parse_dates=True)
    sp500 = pd.read_csv('data/sp500_processed.csv', index_col='Date', parse_dates=True)
    return ftse.dropna(subset=['volatility_annualized']), sp500.dropna(subset=['volatility_annualized'])

@st.cache_resource
def load_models():
    model_ftse = VolatilityLSTM()
    model_ftse.load_state_dict(torch.load('data/model_ftse.pth'))
    model_ftse.eval()

    model_sp500 = VolatilityLSTM()
    model_sp500.load_state_dict(torch.load('data/model_sp500.pth'))
    model_sp500.eval()

    scaler_ftse = joblib.load('data/scaler_ftse.pkl')
    scaler_sp500 = joblib.load('data/scaler_sp500.pkl')

    return model_ftse, model_sp500, scaler_ftse, scaler_sp500

ftse, sp500 = load_data()
model_ftse, model_sp500, scaler_ftse, scaler_sp500 = load_models()

st.title("Volatility Forecasting: GARCH vs LSTM")
st.caption("Comparing classical statistical models against deep learning for market volatility forecasting")

# --- Sidebar controls ---
st.sidebar.header("Controls")
index_choice = st.sidebar.selectbox("Select Index", ["FTSE 100", "S&P 500"])

# Pick the right dataset based on selection
if index_choice == "FTSE 100":
    data = ftse
    color = "#1f4e8c"  # navy
else:
    data = sp500
    color = "#8c1f1f"  # dark red

# --- Historical price and volatility chart ---
st.subheader(f"{index_choice}: Price & Realized Volatility")

col1, col2 = st.columns(2)

with col1:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', 
                                     line=dict(color=color, width=1), name='Close Price'))
    fig_price.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data.index, y=data['volatility_annualized'], mode='lines',
                                   line=dict(color=color, width=1), name='Annualized Volatility'))
    fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- GARCH vs LSTM comparison ---
st.subheader(f"{index_choice}: Model Comparison (Out-of-Sample)")

# Load pre-computed comparison results
if index_choice == "FTSE 100":
    pred_lstm = np.load('data/pred_ftse.npy').flatten()
    actual_lstm = np.load('data/actual_ftse.npy').flatten()
    rmse_lstm = 0.0154
    rmse_garch = 0.0347
else:
    pred_lstm = np.load('data/pred_sp500.npy').flatten()
    actual_lstm = np.load('data/actual_sp500.npy').flatten()
    rmse_lstm = 0.0284
    rmse_garch = 0.0404

col1, col2 = st.columns(2)
col1.metric("LSTM RMSE", f"{rmse_lstm:.4f}", delta=f"{rmse_lstm - rmse_garch:.4f} vs GARCH", delta_color="inverse")
col2.metric("GARCH RMSE", f"{rmse_garch:.4f}")

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(y=actual_lstm, mode='lines', name='Actual', 
                                   line=dict(color='gray', width=2)))
fig_compare.add_trace(go.Scatter(y=pred_lstm, mode='lines', name='LSTM Forecast',
                                   line=dict(color=color, width=1.5)))
fig_compare.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                            xaxis_title="Test Period (trading days)", yaxis_title="Annualized Volatility")
st.plotly_chart(fig_compare, use_container_width=True)

st.info(f"📌 **Key finding:** LSTM achieves {((rmse_garch - rmse_lstm) / rmse_garch * 100):.0f}% lower error than GARCH(1,1) on {index_choice}, "
        f"largely because GARCH reacts noisily to every daily return while LSTM has learned the smoother underlying volatility pattern. "
        f"However, GARCH remains valuable for its interpretability and use in formal risk models (VaR, regulatory capital).")

# --- Live prediction ---
st.subheader(f"{index_choice}: Predict Next-Day Volatility")

st.write("Uses the last 21 trading days of realized volatility to forecast the next day, via the trained LSTM model.")

if st.button(f"🔮 Predict tomorrow's volatility for {index_choice}"):
    # Grab the most recent 21 days of volatility from the full dataset
    recent_vol = data['volatility_annualized'].values[-21:]

    # Select the right model and scaler
    if index_choice == "FTSE 100":
        model, scaler = model_ftse, scaler_ftse
    else:
        model, scaler = model_sp500, scaler_sp500

    # Scale, reshape, and predict — must match training preprocessing exactly
    scaled_input = scaler.transform(recent_vol.reshape(-1, 1)).flatten()
    input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).unsqueeze(-1)  # shape: (1, 21, 1)

    with torch.no_grad():
        model.eval()
        scaled_prediction = model(input_tensor).item()

    prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]
    last_actual = recent_vol[-1]
    change = prediction - last_actual

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Volatility", f"{prediction:.4f}")
    col2.metric("Most Recent Actual", f"{last_actual:.4f}")
    col3.metric("Predicted Change", f"{change:+.4f}")
