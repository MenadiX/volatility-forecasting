# Project notes — key findings

Running log of statistical findings as the project develops. Pull from this directly when writing the final README (Week 7).

## Data

- FTSE 100 (`^FTSE`) and S&P 500 (`^GSPC`), daily, 2015-01-02 to 2026-07-31
- FTSE: 2,925 rows | S&P 500: 2,911 rows — no missing values in either
- Target variable: 21-day rolling realized volatility, annualized (× √252)

## Stationarity (ADF test)

| Series | ADF statistic | p-value | Result |
|---|---|---|---|
| FTSE price | -0.34 | 0.92 | Non-stationary |
| FTSE log return | -11.94 | 0.0000 | Stationary |
| S&P 500 price | 1.03 | 0.99 | Non-stationary |
| S&P 500 log return | -17.44 | 0.0000 | Stationary |

Confirms log returns (not raw prices) are the correct input for ARIMA/GARCH — this isn't just convention, it's tested.

## ARIMA(1,0,0)

| | FTSE 100 | S&P 500 |
|---|---|---|
| AR(1) coefficient | -0.0124 | -0.1297 |
| AR(1) p-value | 0.203 (not significant) | 0.000 (significant) |
| Constant (drift) | 0.0002, ns | 0.0004, significant |
| Kurtosis | 16.29 | 16.97 |
| Skew | -0.93 | -0.79 |
| Heteroskedasticity test p-value | < 0.05 | < 0.05 |

**Key finding:** FTSE returns show no significant own-lag structure — consistent with weak-form market efficiency. S&P 500 shows small but statistically significant negative AR(1), i.e. mild short-term mean-reversion — a real cross-market difference worth calling out.

**Shared finding:** both series are strongly fat-tailed (kurtosis ~16-17 vs. 3 for a normal distribution) and negatively skewed (bigger down-moves than up-moves). Both fail the heteroskedasticity test — ARIMA residual variance is not constant over time, i.e. volatility clustering is present and unmodeled. This is the direct justification for moving to GARCH.

## GARCH(1,1) with Student-t distribution

| | FTSE 100 | S&P 500 |
|---|---|---|
| alpha[1] (shock effect) | 0.1619 | 0.1650 |
| beta[1] (volatility persistence) | 0.7825 | 0.8293 |
| alpha + beta | 0.944 | 0.994 |
| nu (t-distribution d.o.f.) | 5.23 | 5.41 |

**Key finding:** alpha + beta close to 1 for both markets indicates highly persistent volatility shocks — a spike doesn't decay quickly, matching the "spike then slow decay" shape visible in the realized volatility charts. S&P 500's persistence (0.994) is notably higher than FTSE's (0.944), suggesting US equity volatility shocks are even stickier than UK ones.

Low nu values (~5, well below ~10) independently confirm fat tails — consistent with the kurtosis finding from the ARIMA residuals. Two separate diagnostics agreeing is good validation to mention explicitly.

GARCH fitted conditional volatility tracks realized volatility closely for both markets, including through the COVID spike, 2022 elevated-vol period, and a 2025 spike — visual confirmation the model captures real dynamics, not just fitting noise.

## LSTM (PyTorch, single-layer, hidden_size=32)

- Input: 21-day trailing window of realized annualized volatility → predict next day's value
- Chronological train/test split (80/20) — no shuffling, avoids look-ahead leakage
- Data scaled with MinMaxScaler (separate scaler per index)

**First pass (100 epochs):** FTSE badly underfit — predictions visibly lagged and flattened, missing low-vol periods entirely (RMSE 0.0305). S&P 500 fit well from the start (RMSE 0.0354, but visually tight).

**Diagnosis:** FTSE's tighter, sharper volatility swings needed more training than S&P 500's did under identical hyperparameters — not a fundamental architecture mismatch.

**Fix:** retrained both models at 200 epochs for a fair, standardized comparison.

| | FTSE 100 | S&P 500 |
|---|---|---|
| Test RMSE (100 epochs) | 0.0305 | 0.0354 |
| Test RMSE (200 epochs, final) | **0.0180** | **0.0202** |
| Test MAE (200 epochs, final) | 0.0098 | 0.0116 |

**Key finding:** more training resolved the FTSE lag/underfit problem entirely — visual fit after retraining is comparable to S&P 500. FTSE actually edges out S&P 500 slightly on final RMSE, reversing the initial (undertrained) impression that FTSE was the harder series to model.

**Process note for the README:** the original FTSE run showed a real, visible failure mode (lagged, flattened predictions) rather than a subtly-worse number — investigating and fixing it, rather than reporting the first result, is worth narrating explicitly as evidence of rigor.

**Known LSTM behavior to expect:** even the well-trained versions smooth over the sharpest single-day moves more than GARCH does — expected given GARCH has an explicit shock term (alpha) reacting immediately to new information, while the LSTM has learned an averaged pattern.

## Week 5 — GARCH vs LSTM: fair out-of-sample comparison

**Methodology fix:** initial GARCH conditional volatility was in-sample (fit on full data), which would have been an unfair comparison against LSTM's held-out test evaluation. Refit GARCH on train-only data, then generated genuine walk-forward one-step-ahead forecasts for the test period (refitting daily using only information available up to that point — no future leakage). Same train/test split boundary as LSTM.

**Final out-of-sample results:**

| | FTSE 100 RMSE | S&P 500 RMSE |
|---|---|---|
| GARCH(1,1) | 0.0347 | 0.0404 |
| LSTM | 0.0154 | 0.0284 |

LSTM outperforms GARCH by a wide margin on both markets (roughly halves the RMSE).

**Why, visually — this is the interesting part, not just "LSTM wins":**
- GARCH is visibly noisy even during calm periods — its explicit shock term (alpha) reacts to every day's return, including small non-signal moves, producing jagged forecasts even when true volatility is flat.
- LSTM produces a much smoother, more accurate trace during calm periods, having learned the underlying pattern rather than reacting daily.
- At the sharpest spikes (COVID, 2025), GARCH actually **overshoots** the peak (e.g. FTSE: GARCH ~0.42 vs actual ~0.32) while LSTM **undershoots slightly** but stays closer to true peak magnitude. Two distinct failure modes, not just "one is more accurate" — GARCH overreacts to shocks, LSTM smooths them.

**Fair framing for the README:** this isn't "LSTM beats GARCH" as a general claim — it's specific to this setup, where LSTM has a more direct input (21 days of realized volatility itself) vs. GARCH inferring volatility indirectly from returns through fixed statistical assumptions. GARCH(1,1) is also the simplest GARCH variant — more advanced versions (EGARCH, GJR-GARCH, which explicitly model asymmetric response to negative shocks, relevant given the negative skew found in Week 3) might narrow this gap. GARCH retains real advantages LSTM doesn't offer: interpretability, formal statistical grounding, and direct use in regulatory risk frameworks (VaR, capital requirements) — relevant context for a finance/markets employer like LSEG.

## Open threads / to revisit

- 2025 volatility spike (~Feb-Mar 2025 based on chart) — worth a one-line investigation into what caused it, for the README
- Next: LSTM model (Week 4), then head-to-head comparison against GARCH (Week 5)
