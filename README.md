# GARCH(1,1) vs. Fractional Brownian Motion for Financial Forecasting

**Group 4 – Department of Statistics Undergraduate Practitioner Challenge 2024/2025 (LSE)**

Kuan Hui Tan · Chian Yung Chia · Nicholas Tan · Yu See Loi · Renee Robert

---

## Overview

This project evaluates two statistical models widely used in financial time series modelling:

- **GARCH(1,1)** (Generalised AutoRegressive Conditional Heteroskedasticity), which is strong at forecasting volatility and volatility clustering
- **Fractional Brownian Motion (fBM)**, which is strong at capturing long-range dependence

Each model is effective only in specific contexts. GARCH forecasts volatility well but struggles with long-term dependencies, while fBM handles long-range dependence but lags in other areas. This matters in fields like insurance, where macroeconomic and geopolitical risk is managed by estimating the distribution of risk drivers.

We compare both models on the **S&P 500 (SPX)** and **Dow Jones Industrial Average (DJIA)**, then build a **hybrid model** (fBM + GARCH + an adaptive EWMA-momentum drift) that aims to keep the strengths of both.

**Key findings**
- Both models performed as expected within their own strengths: GARCH for volatility detection, fBM for long-term behaviour.
- GARCH(1,1) with rolling estimation tracked realised volatility across the whole test year, suggesting that, used appropriately, it is the more reliable and versatile model.
- The hybrid model achieved a lower MSE than the individual models.

---

## Data

| Item | Detail |
|---|---|
| Source | Yahoo Finance |
| Indices | S&P 500 (SPX), Dow Jones Industrial Average (DJIA) |
| Period | 1 Jan 2000 – 31 Dec 2024 |
| Training period | 2000 – 2023 |
| Test period | 2024 |
| Returns | Logarithmic returns (daily for GARCH, monthly for fBM) |

**Why different time scales?** GARCH relies on volatility clustering, which shows up most clearly in high-frequency, short-horizon data, so it uses **daily** returns. fBM models continuous, long-memory processes, so it uses **monthly** returns. Lower frequency also removes short-term noise and market microstructure effects, giving a more reliable estimate of the Hurst exponent (H).

### Summary Statistics (Daily Log Returns)

| Statistic | SPX | DJIA |
|---|---|---|
| Count | 6,287 | 6,287 |
| Mean | 0.000223 | 0.000210 |
| Std | 0.012235 | 0.011635 |
| Min | -0.127652 | -0.138418 |
| 25% | -0.004806 | -0.004586 |
| 50% | 0.000602 | 0.000505 |
| 75% | 0.005903 | 0.005585 |
| Max | 0.109572 | 0.107643 |
| Skew | -0.386206 | -0.371867 |
| Kurtosis | 10.403989 | 12.911276 |

Both indices have a similar positive mean drift, but negative skew shows losses occur more often. SPX has slightly higher volatility and lower kurtosis than DJIA, meaning more frequent but more stable fluctuations.

---

## Getting Started

All analysis was done in Python, mainly with the [`arch`](https://arch.readthedocs.io/), [`statsmodels`](https://www.statsmodels.org/) and [`nolds`](https://nolds.readthedocs.io/) packages.

```bash
pip install numpy pandas scipy matplotlib yfinance arch statsmodels nolds
```

Yahoo Finance tickers: `^GSPC` (S&P 500) and `^DJI` (Dow Jones).

---

## Methodology

### GARCH(1,1)

1. **Stationarity check.** GARCH assumes stable statistical properties, especially variance. The Augmented Dickey-Fuller test confirms both series are stationary:

   | Index | ADF Statistic | p-value |
   |---|---|---|
   | SPX | -19.1006 | 0.0000 |
   | DJIA | -19.1775 | 0.0000 |

   Critical values: 1% = -3.431, 5% = -2.862, 10% = -2.567.

2. **Model selection.** Alternative GARCH(p,q) specifications were compared on the training period. GARCH(1,1) was chosen for its simpler computation, since higher orders gave only negligible improvements.

   | Index | Model | μ | ω | α₁ | α₂ | β₁ | β₂ | Log-lik | AIC | BIC |
   |---|---|---|---|---|---|---|---|---|---|---|
   | SPX | GARCH(1,1) | 0.0772 | 0.0143 | 0.1206 | – | 0.8754 | – | -8519.97 | 17049.94 | 17083.68 |
   | SPX | GARCH(2,1) | 0.0775 | 0.0206 | 0.0591 | 0.0912 | 0.8431 | – | -8511.15 | 17034.30 | 17074.78 |
   | SPX | GARCH(1,2) | 0.0772 | 0.0143 | 0.1206 | – | 0.8754 | 0.0000 | -8519.97 | 17051.94 | 17092.42 |
   | SPX | GARCH(2,2) | 0.0778 | 0.0234 | 0.0595 | 0.1141 | 0.6430 | 0.1759 | -8518.80 | 17035.60 | 17082.83 |
   | DJIA | GARCH(1,1) | 0.0682 | 0.0153 | 0.1196 | – | 0.8729 | – | -8199.00 | 16408.01 | 16441.74 |
   | DJIA | GARCH(2,1) | 0.0682 | 0.0199 | 0.0668 | 0.0745 | 0.8484 | – | -8192.97 | 16397.93 | 16438.41 |
   | DJIA | GARCH(1,2) | 0.0682 | 0.0153 | 0.1196 | – | 0.8729 | 0.0000 | -8199.00 | 16410.01 | 16450.49 |
   | DJIA | GARCH(2,2) | 0.0685 | 0.0238 | 0.0671 | 0.1044 | 0.5874 | 0.2288 | -8192.54 | 16399.09 | 16446.31 |

3. **Residual distribution.** In-sample standardised residuals follow a Student's t-distribution, with εₜ ~ t₆.₄ for SPX and εₜ ~ t₆.₅ for DJIA. These fat tails reflect the frequent extreme returns typical of financial data.

4. **Mean drift.** Using yₜ = xₜ′b + εₜ, the estimated mean drift is close to zero for both indices (SPX: 0.0772, DJIA: 0.0682), so it is treated as negligible in later calculations.

5. **Independence.** The autocorrelation function of the standardised residuals is minimal after the first lag, consistent with independent increments and white noise.

6. **Forecasting.** Realised volatility was forecast over **5, 22 and 252 trading days**, using both **static** and **rolling** (21-day window) estimation, to capture behaviour in both stable and dynamic markets.

7. **Evaluation.** Forecast accuracy metrics, the QLIKE loss function and Mincer-Zarnowitz regression.

### Fractional Brownian Motion

1. **Cumulative log returns** were used alongside regular log returns, since they show long-term dependencies more clearly.
2. **Hurst exponent estimation** used two methods, **Detrended Fluctuation Analysis (DFA)** and **Rescaled Range (R/S) analysis**, combined with Monte Carlo simulation to reduce discrepancies.
3. **Static vs. rolling estimation:** both were run for each method. They gave very similar distributions, so the static approach was chosen as the primary method for its simplicity.
4. Unlike GARCH, fBM is built on **dependent increments**, which is what gives it long-range dependence.
5. **Forecasting** extrapolated the time series over one year using DFA and R/S.
6. **Evaluation** used the same tests as GARCH, plus a **2-sample Kolmogorov-Smirnov test** for scale invariance.

### Hybrid Model: fBM + GARCH + EWMA-Momentum Drift

The hybrid combines GARCH(1,1) and fBM with an **adaptive drift** that blends an EWMA-based drift with market-adjusted momentum. The goal is to monitor short-term volatility while also capturing long-term trends, with a drift that evolves with realistic market behaviour.

---

## Results

### GARCH(1,1)

**Rolling forecasts.** The 1-step predicted volatility followed the broad trends of realised 21-day rolling volatility throughout 2024 for both indices, ranging roughly between 0.4% and 1.6%. It lagged slightly during sudden spikes and dips, but correctly forecast the clustered volatility that follows high-volatility periods.

Rolling 1-step forecast accuracy (DJIA):

| Metric | Value |
|---|---|
| MSE (volatility) | 0.014130 |
| QLIKE (approx.) | 0.226043 |
| Mincer-Zarnowitz R² | 0.696 |
| MZ intercept | 0.0322 (p = 0.251) |
| MZ slope | 0.8753 (p < 0.001) |

The intercept is not significantly different from zero and the slope is close to 1, indicating a reasonably unbiased forecast.

**Static forecasts.** These fit well over short and medium horizons but drift away from realised volatility over the long horizon.

| Horizon | SPX MSE | DJIA MSE |
|---|---|---|
| Short (5 days) | 0.187339 | 0.157046 |
| Medium (22 days) | 0.028586 | 0.050855 |
| Long (252 days) | 0.321438 | 0.284740 |

### Fractional Brownian Motion

Static and rolling estimates for both R/S and DFA were close together and followed the general upward pattern of actual cumulative log returns. They underestimated the level, possibly because averaging the Monte Carlo paths neutralised the indices' mean drift.

| Index | Method | Hurst (H) | Train RMSE | Test RMSE | Test MAPE (%) | KS p (Test) |
|---|---|---|---|---|---|---|
| DJIA | Rolling DFA (win = 120) | 0.4823 | 0.3313 | 0.0528 | 87.31 | 0.1812 |
| DJIA | Rolling R/S (win = 120) | 0.6802 | 0.3567 | 0.0545 | 87.89 | 0.1778 |
| DJIA | Static DFA | 0.5548 | 0.3232 | 0.0562 | 86.68 | 0.1675 |
| DJIA | Static R/S | 0.6042 | 0.3411 | 0.0548 | 88.29 | 0.1825 |
| SPX | Rolling DFA (win = 120) | 0.4622 | 0.4504 | 0.1179 | 96.22 | 0.1496 |
| SPX | Rolling R/S (win = 120) | 0.7134 | 0.4740 | 0.1037 | 98.89 | 0.2038 |
| SPX | Static DFA | 0.5942 | 0.4498 | 0.1074 | 90.35 | 0.1974 |
| SPX | Static R/S | 0.5944 | 0.4463 | 0.1112 | 92.65 | 0.1793 |

For SPX, static DFA and R/S give the same Hurst exponent (**H ≈ 0.594**), implying self-similarity across time horizons. H > 0.5 indicates persistent, trending behaviour (long memory).

### Hybrid Model

| Index | Hurst Exponent | Dynamic Monthly Log Drift | MSE |
|---|---|---|---|
| SPX | 0.5944 | 0.013392 | 0.011918 |
| DJIA | 0.6042 | 0.009078 | 0.005164 |

The mean Monte Carlo path follows the upward trend of actual 2024 cumulative log returns, with a lower MSE than the individual models.

---

## Conclusion

The two models capture different aspects of financial data, which makes a direct comparison difficult. Both performed as expected within their own strengths. GARCH(1,1)'s smoother forecasts cannot perfectly match daily volatility swings, but with rolling estimation it still showed some ability to predict over the long term, possibly outperforming fBM even for long-range dependence. This suggests that **GARCH, used appropriately, is generally the more reliable and versatile model**.

The **hybrid fBM + GARCH + EWMA-momentum drift model** keeps short-term volatility monitoring while supporting long-term trends. Its drift reflects realistic market conditions, giving a lower MSE than either model alone and better capturing market risk. Further refinement could improve future log return predictions.
