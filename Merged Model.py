import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from fbm import FBM
from arch import arch_model
from scipy.stats import laplace
from hurst import compute_Hc  # Hurst exponent calculation
from numpy import mean, square

#SP500

# **Fetch historical S&P 500 data**
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-02-18', interval='1mo', auto_adjust=True)

# Ensure index is in datetime format
sp500.index = pd.to_datetime(sp500.index)

# **Calculate Monthly Log Returns**
sp500['Log_Returns'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
monthly_returns = sp500['Log_Returns'].dropna()

# **Updated Training and Testing Periods**
training_period = monthly_returns.loc['2000-01-01':'2023-12-31']  # Training until end of 2023
testing_period = monthly_returns.loc['2024-01-01':]  # Testing starts in 2024

# **Compute Hurst Exponent from Historical Prices**
if len(training_period) < 20:
    print("Warning: Not enough data points for Hurst exponent calculation. Using default H = 0.5.")
    H_optimal = 0.5
else:
    try:
        H_optimal, _, _ = compute_Hc(training_period.values, kind="change", simplified=True)
    except FloatingPointError:
        print("Warning: Invalid values encountered in log10 for Hurst exponent calculation. Using default H = 0.5.")
        H_optimal = 0.5

print(f"Calculated Hurst Exponent for S&P 500: {H_optimal:.4f}")

# **Hybrid Drift Calculation: EWMA-Based with Momentum Adjustment**
ewma_drift = training_period.ewm(span=60).mean().iloc[-1]
recent_momentum = np.mean(training_period[-12:])  # Last 12 months mean return
momentum_threshold = 0.005  # Adjust drift if momentum is significant

if abs(recent_momentum) > momentum_threshold:
    adjusted_drift = (ewma_drift + recent_momentum) / 2  # Blend EWMA and Momentum
else:
    adjusted_drift = ewma_drift

dynamic_drift = adjusted_drift

print(f"Dynamic Monthly Log Drift (Hybrid EWMA & Momentum-Based): {dynamic_drift:.6f}")

# **Fit GARCH(1,1) Model for Better Volatility Capture**
scaled_testing_returns = testing_period * 100  # Scale for better GARCH performance
garch_model = arch_model(scaled_testing_returns, mean="Constant", vol="Garch", p=1, q=1, dist="t")
garch_fit = garch_model.fit(disp="off")

garch_volatility = garch_fit.conditional_volatility / 100  # Undo scaling
garch_volatility = garch_volatility.iloc[1:].values  # Drop first value to match fBM

# **Monte Carlo Simulation Parameters**
num_simulations = 10000
num_steps = len(testing_period)

# Fetch last known price from training period
last_train_date = training_period.index[-1]
last_train_price = sp500['Close'].loc[sp500.index.asof(last_train_date)]

if isinstance(last_train_price, pd.Series):
    last_train_price = last_train_price.iloc[0]

# **Monte Carlo Storage**
simulated_price_paths = np.zeros((num_simulations, num_steps))

for i in range(num_simulations):
    f = FBM(n=num_steps, hurst=H_optimal, length=1, method='daviesharte')
    fbm_path = f.fbm()
    fbm_increments = np.diff(fbm_path)
    
    fbm_increments = (fbm_increments - np.mean(fbm_increments)) / np.std(fbm_increments)
    
    min_length = min(len(fbm_increments), len(garch_volatility))
    fbm_increments = fbm_increments[:min_length]
    garch_volatility_adj = garch_volatility[:min_length]

    garch_fbm_returns = dynamic_drift + fbm_increments * garch_volatility_adj

    garch_fbm_cum_returns = np.concatenate(([0], np.cumsum(garch_fbm_returns)))
    
    simulated_prices = garch_fbm_cum_returns
    simulated_price_paths[i, :min_length + 1] = simulated_prices

mean_simulation = np.mean(simulated_price_paths, axis=0)

# **Calculate Mean Squared Error (MSE) Between Actual and Simulated Log Returns**
actual_cum_log_returns = np.log(sp500['Close'].loc[testing_period.index] / last_train_price)
mse = mean(square(actual_cum_log_returns.values[:len(mean_simulation)] - mean_simulation))
print(f"Mean Squared Error (MSE): {mse:.6f}")

plt.figure(figsize=(14, 7))
plt.plot(testing_period.index, mean_simulation, label="Mean Monte Carlo Simulation", color="red", linewidth=2, linestyle='dashed')
plt.plot(testing_period.index, actual_cum_log_returns, label="Actual S&P 500 Cumulative Log Returns", color="blue", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Cumulative Log Returns")
plt.title("Monte Carlo Simulations of S&P 500 - Cumulative Log Returns using fBM + GARCH + Hybrid EWMA-Momentum Drift")
plt.legend()
plt.grid()
plt.show()




# DJIA

# **Fetch historical DJIA data**
djia = yf.download('^DJI', start='2000-01-01', end='2025-02-18', interval='1mo', auto_adjust=True)

# Ensure index is in datetime format
djia.index = pd.to_datetime(djia.index)

# **Calculate Monthly Log Returns**
djia['Log_Returns'] = np.log(djia['Close'] / djia['Close'].shift(1))
monthly_returns = djia['Log_Returns'].dropna()

# **Updated Training and Testing Periods**
training_period = monthly_returns.loc['2000-01-01':'2023-12-31']  # Training until end of 2023
testing_period = monthly_returns.loc['2024-01-01':]  # Testing starts in 2024

# **Compute Hurst Exponent from Historical Prices**
if len(training_period) < 20:
    print("Warning: Not enough data points for Hurst exponent calculation. Using default H = 0.5.")
    H_optimal = 0.5
else:
    try:
        H_optimal, _, _ = compute_Hc(training_period.values, kind="change", simplified=True)
    except FloatingPointError:
        print("Warning: Invalid values encountered in log10 for Hurst exponent calculation. Using default H = 0.5.")
        H_optimal = 0.5

print(f"Calculated Hurst Exponent for DJIA: {H_optimal:.4f}")

# **Hybrid Drift Calculation: EWMA-Based with Momentum Adjustment**
ewma_drift = training_period.ewm(span=60).mean().iloc[-1]
recent_momentum = np.mean(training_period[-12:])  # Last 12 months mean return
momentum_threshold = 0.005  # Adjust drift if momentum is significant

if abs(recent_momentum) > momentum_threshold:
    adjusted_drift = (ewma_drift + recent_momentum) / 2  # Blend EWMA and Momentum
else:
    adjusted_drift = ewma_drift

dynamic_drift = adjusted_drift

print(f"Dynamic Monthly Log Drift (Hybrid EWMA & Momentum-Based): {dynamic_drift:.6f}")

# **Fit GARCH(1,1) Model for Better Volatility Capture**
scaled_testing_returns = testing_period * 100  # Scale for better GARCH performance
garch_model = arch_model(scaled_testing_returns, mean="Constant", vol="Garch", p=1, q=1, dist="t")
garch_fit = garch_model.fit(disp="off")

garch_volatility = garch_fit.conditional_volatility / 100  # Undo scaling
garch_volatility = garch_volatility.iloc[1:].values  # Drop first value to match fBM

# **Monte Carlo Simulation Parameters**
num_simulations = 10000
num_steps = len(testing_period)

# Fetch last known price from training period
last_train_date = training_period.index[-1]
last_train_price = djia['Close'].loc[djia.index.asof(last_train_date)]

if isinstance(last_train_price, pd.Series):
    last_train_price = last_train_price.iloc[0]

# **Monte Carlo Storage**
simulated_price_paths = np.zeros((num_simulations, num_steps))

for i in range(num_simulations):
    f = FBM(n=num_steps, hurst=H_optimal, length=1, method='daviesharte')
    fbm_path = f.fbm()
    fbm_increments = np.diff(fbm_path)
    
    fbm_increments = (fbm_increments - np.mean(fbm_increments)) / np.std(fbm_increments)
    
    min_length = min(len(fbm_increments), len(garch_volatility))
    fbm_increments = fbm_increments[:min_length]
    garch_volatility_adj = garch_volatility[:min_length]

    garch_fbm_returns = dynamic_drift + fbm_increments * garch_volatility_adj

    garch_fbm_cum_returns = np.concatenate(([0], np.cumsum(garch_fbm_returns)))
    
    simulated_prices = garch_fbm_cum_returns
    simulated_price_paths[i, :min_length + 1] = simulated_prices

mean_simulation = np.mean(simulated_price_paths, axis=0)

# **Calculate Mean Squared Error (MSE) Between Actual and Simulated Log Returns**
actual_cum_log_returns = np.log(djia['Close'].loc[testing_period.index] / last_train_price)
mse = mean(square(actual_cum_log_returns.values[:len(mean_simulation)] - mean_simulation))
print(f"Mean Squared Error (MSE): {mse:.6f}")

plt.figure(figsize=(14, 7))
plt.plot(testing_period.index, mean_simulation, label="Mean Monte Carlo Simulation", color="red", linewidth=2, linestyle='dashed')
plt.plot(testing_period.index, actual_cum_log_returns, label="Actual DJIA Cumulative Log Returns", color="blue", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Cumulative Log Returns")
plt.title("Monte Carlo Simulations of DJIA - Cumulative Log Returns using fBM + GARCH + Hybrid EWMA-Momentum Drift")
plt.legend()
plt.grid()
plt.show()
