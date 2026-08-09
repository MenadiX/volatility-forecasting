# Volatility Forecasting: GARCH vs LSTM

Forecasting stock market volatility on the FTSE 100 and S&P 500, comparing a classical econometric model (GARCH) against a deep learning model (LSTM) — with a fair, out-of-sample evaluation and an interactive dashboard.


## The question

Which approach forecasts near-term market volatility better: a purpose-built statistical model with 40 years of academic backing, or a neural network that simply learns patterns from data? And when each one is wrong, *how* is it wrong — do they fail the same way?

## Key finding

LSTM outperformed GARCH(1,1) on both indices, roughly halving the out-of-sample RMSE (FTSE 100: 0.0154 vs 0.0347 · S&P 500: 0.0284 vs 0.0404). But the more interesting result is *how* each model fails:

- **GARCH is noisy but reactive** — its shock term responds to every day's return, even small non-signal moves, producing jagged forecasts during genuinely calm periods. At true volatility spikes, it tends to **overshoot** the peak.
- **LSTM is smooth but lagging** — it learns the underlying pattern well and tracks calm periods cleanly, but tends to **undershoot** the sharpest spikes, since it's optimizing for an averaged pattern rather than reacting instantly.

This isn't a case of one model being categorically better — it's two different failure modes suited to different uses. GARCH remains the standard in finance for good reason: it's interpretable, statistically grounded, and used directly in regulatory risk frameworks (VaR, capital requirements). LSTM wins on raw forecast accuracy here, but offers none of that interpretability.

![GARCH vs LSTM comparison chart](assets/garch_vs_lstm_comparison.png)

## Methodology

**Data:** Daily FTSE 100 (`^FTSE`) and S&P 500 (`^GSPC`) prices, 2015–2026, via `yfinance`. No missing values.

**Preprocessing:**
- Log returns computed and confirmed stationary via ADF test (prices are not — p ≈ 0.92–0.99; returns are — p ≈ 0.0000)
- 21-day rolling realized volatility (annualized via ×√252) used as the common target variable for both models

**GARCH(1,1) with Student-t distribution:**
- Chosen after confirming heavy tails in the data (kurtosis ~16–17 vs. 3 for a normal distribution)
- alpha + beta ≈ 0.94–0.99 for both markets → highly persistent volatility shocks, consistent with the spike-then-slow-decay shape seen in realized volatility
- Evaluated via **genuine walk-forward forecasting**: refit daily using only information available up to that point, never given access to test-period data

**LSTM (PyTorch):**
- Single-layer LSTM, hidden size 32, trained on 21-day trailing windows of realized volatility to predict the next day
- Chronological train/test split (no shuffling — avoids look-ahead leakage)
- 200 training epochs (a first pass at 100 epochs visibly underfit FTSE specifically — see notes below)

**Fair comparison:** both models are evaluated on identical, held-out test periods using genuinely out-of-sample forecasts — not in-sample fit, which would have made GARCH look artificially stronger.

## A debugging note worth mentioning

The first LSTM run (100 epochs) fit S&P 500 well but visibly underfit FTSE 100 — predictions lagged and flattened, missing low-volatility periods entirely. Rather than accept the first result, I investigated: retraining for 200 epochs resolved it completely (FTSE RMSE improved from 0.0305 to 0.0154), and the same retrain improved S&P 500 too, so both models were standardized at 200 epochs for a fair comparison. The original "FTSE is harder to model" impression was an artifact of undertraining, not a real market difference.

## Repository structure

```
├── 01_data_collection.ipynb      # Data pull, EDA, stationarity testing
├── 02_arima_garch_models.ipynb   # ARIMA baseline, GARCH(1,1) fitting
├── 03_lstm_model.ipynb           # LSTM windowing, training, evaluation
├── 04_comparison.ipynb           # Fair out-of-sample GARCH vs LSTM comparison
├── app.py                        # Streamlit dashboard
├── data/                         # Processed datasets, saved models, scalers
├── requirements.txt
└── NOTES.md                      # Full running log of statistical findings
```

## Running it locally

```bash
git clone <repo-url>
cd volatility-forecasting
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

Processed data and trained model weights are included in the repo, so the dashboard runs immediately without needing to retrain or re-pull data.

## What I'd explore next

- More advanced GARCH variants (EGARCH, GJR-GARCH) to explicitly model the asymmetric response to negative shocks suggested by the negative skew in the return data
- A hybrid approach — using GARCH's volatility estimate as an additional LSTM input feature
- Extending to a longer forecast horizon (5-day, 21-day ahead) rather than just next-day

## Tech stack

Python · PyTorch · statsmodels · arch · Streamlit · Plotly · pandas · yfinance
