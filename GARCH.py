import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

# Define the tickers and the date range for indices
tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI"}
start_date = "2000-01-01"
end_date = "2024-12-31"

# Create an empty dictionary to store the data for each index
stocks_data = {}

# Loop over each ticker and download the data (daily data ensured by interval="1d")
for name, ticker in tickers.items():
    print(f"Downloading data for {name} ({ticker})...")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", actions=True)
    stocks_data[name] = data
    print(f"Data for {name}:")
    print(data.head(), "\n")


# Step 1: Plot Index Price Trend (Before Computing Log Returns)
def plot_stock_trends(stocks_data):
    plt.figure(figsize=(12, 6))
    for name, df in stocks_data.items():
        # Use 'Adj Close' if available; otherwise, 'Close'
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        plt.plot(df.index, df[col], label=name)
    plt.xlabel("Year")
    plt.ylabel("Price (USD)")
    plt.title("Index Price Trend (2000 - 2024)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function to plot index price trends
plot_stock_trends(stocks_data)


# Step 2: Compute the log returns for S&P 500 and Dow Jones
def process_data_daily(df: pd.DataFrame) -> pd.Series:
    # Handle MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Choose price column: 'Adj Close' if available; otherwise, 'Close'
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    daily_prices = df[price_col]

    # Compute daily log returns: ln(P_t / P_t-1)
    daily_log_returns = np.log(daily_prices / daily_prices.shift(1))
    daily_log_returns.dropna(inplace=True)

    return daily_log_returns


# Process and print cleaned daily log returns for each index
for name, df in stocks_data.items():
    returns_daily = process_data_daily(df)
    print(f"Cleaned daily log returns for {name}:")
    print(returns_daily.head(), "\n")

# Step 3: Basic Data Analysis for Daily Log Returns for S&P 500 and Dow Jones
summary_stats_daily = {}
for name, df in stocks_data.items():
    returns_daily = process_data_daily(df)
    desc_daily = returns_daily.describe()  # Summary statistics
    desc_daily["skew"] = returns_daily.skew()  # Skewness
    desc_daily["kurtosis"] = returns_daily.kurt()  # Kurtosis
    summary_stats_daily[name] = desc_daily

# Combine the summary stats into a single DataFrame for a neat summary table
daily_summary_table = pd.DataFrame(summary_stats_daily)
print("Summary Table for Daily Log Returns:")
print(daily_summary_table)

# --- Plotting Histograms and Q-Q Plots for S&P 500 and Dow Jones ---
assets = ["S&P 500", "Dow Jones"]

for asset in assets:
    # Get daily returns for the asset
    returns = process_data_daily(stocks_data[asset])

    # Check that returns is not empty before proceeding
    if returns.empty:
        print(f"No returns data available for {asset}. Skipping distribution analysis.")
        continue

    # Estimate parameters for Normal Distribution
    mu, sigma = returns.mean(), returns.std()

    # Fit t distribution using MLE; returns: (df, loc, scale)
    try:
        df_t, loc_t, scale_t = stats.t.fit(returns)
    except Exception as e:
        print(f"Error fitting t distribution for {asset}: {e}")
        continue

    # Create x-axis values for PDFs (cover the range of the data)
    x = np.linspace(returns.min(), returns.max(), 200)

    # Compute the theoretical PDFs
    pdf_norm = stats.norm.pdf(x, loc=mu, scale=sigma)
    pdf_t = stats.t.pdf(x, df_t, loc=loc_t, scale=scale_t)

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Distribution Analysis for {asset}", fontsize=16)

    # --- Histogram with Normal PDF Overlay ---
    ax = axes[0, 0]
    ax.hist(returns, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    ax.plot(x, pdf_norm, 'r-', lw=2, label='Normal PDF')
    ax.set_title("Histogram with Normal PDF")
    ax.legend()

    # --- Histogram with t PDF Overlay ---
    ax = axes[0, 1]
    ax.hist(returns, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
    ax.plot(x, pdf_t, 'b-', lw=2, label='t PDF')
    ax.set_title("Histogram with t PDF")
    ax.legend()

    # --- Q-Q Plot vs. Normal Distribution ---
    ax = axes[1, 0]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot vs. Normal Distribution")

    # --- Q-Q Plot vs. t Distribution ---
    ax = axes[1, 1]
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    # Compute plotting positions
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = stats.t.ppf(probs, df_t, loc=loc_t, scale=scale_t)

    ax.plot(theoretical_quantiles, sorted_returns, 'o', markersize=3, label='Data')
    # Plot 45-degree line for reference
    min_val = min(theoretical_quantiles[0], sorted_returns[0])
    max_val = max(theoretical_quantiles[-1], sorted_returns[-1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='45-degree line')
    ax.set_title("Q-Q Plot vs. t Distribution")
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Step 4: Check for Stationarity ---
def check_stationarity(stocks_data):
    print("\nChecking Stationarity for All Indices\n")

    fig, axes = plt.subplots(1, len(stocks_data), figsize=(14, 5), sharey=True)

    for idx, (name, df) in enumerate(stocks_data.items()):
        returns = process_data_daily(df)
        rolling_mean = returns.rolling(window=21).mean()
        rolling_std = returns.rolling(window=21).std()

        axes[idx].plot(returns, color='blue', label="Daily Log Returns", alpha=0.7)
        axes[idx].plot(rolling_mean, color='red', label="Rolling Mean (21 days)", linestyle="--")
        axes[idx].plot(rolling_std, color='green', label="Rolling Std (21 days)", linestyle=":")

        axes[idx].set_title(f"Rolling Statistics for {name} (Daily Data)")
        axes[idx].set_xlabel("Date")
        if idx == 0:
            axes[idx].set_ylabel("Log Returns")
        axes[idx].legend()
        axes[idx].grid()

        # Perform the Augmented Dickey-Fuller (ADF) Test
        adf_result = adfuller(returns.dropna())
        print(f"\nADF Test Results for {name}:")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Critical Values: {adf_result[4]}")
        if adf_result[1] < 0.05:
            print(f"✅ {name} series is stationary (p-value < 0.05).")
        else:
            print(f"❌ {name} series is non-stationary (p-value >= 0.05).")

    plt.tight_layout()
    plt.show()


# Run the stationarity test for both indices
check_stationarity(stocks_data)






# --- Step 5: Fit Multiple GARCH Variants for Each Index (Daily Data) ---
import pandas as pd
from arch import arch_model

# --- Step 5: Fit Multiple GARCH Variants for Each Index (Daily Data) ---
garch_model_variants = {
    "GARCH(1,1)": {"p": 1, "q": 1},
    "GARCH(2,1)": {"p": 2, "q": 1},
    "GARCH(1,2)": {"p": 1, "q": 2},
    "GARCH(2,2)": {"p": 2, "q": 2},
}

garch_models = {}  # {index_name: {model_label: model}}
garch_fits = {}    # {index_name: {model_label: fit}}
summary_records = []  # To store summary info for S&P 500 and DowJones

# Define the names for which you want to include results in the summary table.
summary_indices = {"S&P 500", "DowJones", "Dow Jones"}

for name, df in stocks_data.items():
    print(f"\nFitting GARCH models for {name}...")
    returns = process_data_daily(df)

    if returns.empty:
        print(f"No returns data available for {name}. Skipping GARCH fitting.")
        continue

    garch_models[name] = {}
    garch_fits[name] = {}

    for model_label, params in garch_model_variants.items():
        try:
            model = arch_model(returns, vol="Garch", p=params["p"], q=params["q"], dist='t', rescale=True)
            fit = model.fit(disp="off")
            garch_models[name][model_label] = model
            garch_fits[name][model_label] = fit

            # Extract important parameters: mu, omega, alphas, and betas
            fit_params = fit.params
            mu = fit_params.get("mu", None)
            omega = fit_params.get("omega", None)

            # Extract alpha and beta parameters (names starting with "alpha" or "beta")
            alpha_keys = [key for key in fit_params.index if key.lower().startswith("alpha")]
            beta_keys = [key for key in fit_params.index if key.lower().startswith("beta")]

            alpha_keys.sort()
            beta_keys.sort()

            # Prepare labeled alpha and beta values (up to 2 each)
            alpha_vals = {}
            beta_vals = {}
            for i in range(2):
                if i < len(alpha_keys):
                    alpha_vals[f"alpha{i+1}"] = f"{fit_params[alpha_keys[i]]:.4f}"
                else:
                    alpha_vals[f"alpha{i+1}"] = None
            for i in range(2):
                if i < len(beta_keys):
                    beta_vals[f"beta{i+1}"] = f"{fit_params[beta_keys[i]]:.4f}"
                else:
                    beta_vals[f"beta{i+1}"] = None

            # Loglikelihood, AIC, and BIC
            loglik = fit.loglikelihood
            aic = fit.aic
            bic = fit.bic

            # Print the important parameters for clarity
            print(f"\nImportant Parameters for {name} - {model_label}:")
            print(f"  mu: {mu:.4f}" if mu is not None else "  mu: N/A")
            print(f"  omega: {omega:.4f}" if omega is not None else "  omega: N/A")
            for key in sorted(alpha_vals.keys()):
                print(f"  {key}: {alpha_vals[key] if alpha_vals[key] is not None else 'N/A'}")
            for key in sorted(beta_vals.keys()):
                print(f"  {key}: {beta_vals[key] if beta_vals[key] is not None else 'N/A'}")
            print(f"  Loglikelihood: {loglik:.4f}")
            print(f"  AIC: {aic:.4f}")
            print(f"  BIC: {bic:.4f}")

            # Save the results in the summary table if the index is S&P 500 or DowJones
            if name in summary_indices:
                record = {
                    "Index": name,
                    "Model": model_label,
                    "mu": f"{mu:.4f}" if mu is not None else None,
                    "omega": f"{omega:.4f}" if omega is not None else None,
                    "alpha1": alpha_vals.get("alpha1"),
                    "alpha2": alpha_vals.get("alpha2"),
                    "beta1": beta_vals.get("beta1"),
                    "beta2": beta_vals.get("beta2"),
                    "Loglik": f"{loglik:.4f}",
                    "AIC": f"{aic:.4f}",
                    "BIC": f"{bic:.4f}"
                }
                summary_records.append(record)

        except Exception as e:
            print(f"Error fitting {model_label} for {name}: {e}")

# Create and print the summary table for S&P 500 and DowJones with spaced out columns
if summary_records:
    summary_df = pd.DataFrame(summary_records)
    print("\nSummary Table for S&P 500 and DowJones:")
    print(summary_df.to_string(index=False, col_space=12))
else:
    print("\nNo summary records available for S&P 500 and DowJones.")

# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None

# Step 6: Extract Residuals for S&P 500 and DowJones from the Fitted GARCH Models (Daily Data)
residuals = {}       # Raw residuals: εₜ = rₜ - μ (daily)
std_residuals = {}   # Standardized residuals: zₜ = εₜ / σₜ (daily)

# For each asset, select a specific model fit.
# If multiple models were fitted (i.e. a dict), choose "GARCH(1,1)" if available,
# otherwise use the first available model.
for name, fits in garch_fits.items():
    if isinstance(fits, dict):
        chosen_fit = fits["GARCH(1,1)"] if "GARCH(1,1)" in fits else list(fits.values())[0]
    else:
        chosen_fit = fits
    residuals[name] = chosen_fit.resid         # Raw residuals
    std_residuals[name] = chosen_fit.std_resid    # Standardized residuals

# Determine the keys for S&P 500 and Dow Jones from the residuals dictionary
sp_key = get_index_key(["S&P 500", "SP500", "S&P500"], residuals)
dj_key = get_index_key(["DowJones", "Dow Jones", "DJ"], residuals)

if sp_key is None:
    raise KeyError("S&P 500 key not found in residuals dictionary!")
if dj_key is None:
    raise KeyError("DowJones key not found in residuals dictionary!")

# Create DataFrames to show the first 5 values side by side for S&P 500 and Dow Jones
raw_resid_df = pd.DataFrame({
    "S&P 500": residuals[sp_key].head(),
    "Dow Jones": residuals[dj_key].head()
})

std_resid_df = pd.DataFrame({
    "S&P 500": std_residuals[sp_key].head(),
    "Dow Jones": std_residuals[dj_key].head()
})

print("📊 Side-by-side Raw Residuals (Daily):")
print(raw_resid_df)
print("\n📊 Side-by-side Standardized Residuals (Daily):")
print(std_resid_df)

# Calculate the mean difference (actual returns - raw residuals) for each index
assets = [sp_key, dj_key]
mean_diff = {}  # To store the final mean difference for each asset

for asset in assets:
    # Get the actual returns (in the same units as used in the GARCH fit) and convert to percent
    actual_returns = process_data_daily(stocks_data[asset]) * 100

    # Choose the fitted model for the asset (preferably "GARCH(1,1)")
    fits = garch_fits[asset]
    chosen_fit = fits["GARCH(1,1)"] if "GARCH(1,1)" in fits else list(fits.values())[0]
    raw_resid = chosen_fit.resid

    # Compute the difference: actual_returns - raw_resid. This difference should equal the estimated μ.
    diff_series = actual_returns - raw_resid
    print(diff_series)

    # Compute the mean of these differences (should be very close to the estimated mean return μ)
    mean_diff_value = diff_series.mean()
    mean_diff[asset] = mean_diff_value
    print(f"Mean difference (actual returns - raw residuals) for {asset}: {mean_diff_value:.6f}")

print("\nFinal Mean Differences:")
print(mean_diff)

# Step 7: Visualizing the Distribution of Daily Standardized Residuals (Histogram & Q-Q Plot)
# For each index in our standardized residuals, we overlay the theoretical t-distribution (from the fitted model)
# instead of a KDE curve, and adjust the Q-Q plot accordingly.

# We'll iterate over the indices available in std_residuals
for name in std_residuals:
    # Retrieve the corresponding fitted model.
    # If multiple models were fitted, we select "GARCH(1,1)" if available, otherwise the first model.
    fits = garch_fits[name]
    chosen_fit = fits["GARCH(1,1)"] if "GARCH(1,1)" in fits else list(fits.values())[0]
    # Get the degrees of freedom from the fitted parameters. Default to 10 if not available.
    nu = chosen_fit.params.get("nu", 10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Left Subplot: Histogram with Theoretical t Distribution Overlay -----
    # Plot histogram (density scaled) without the built-in KDE.
    sns.histplot(
        std_residuals[name],
        bins=30,
        stat="density",  # y-axis as density
        kde=False,
        edgecolor="black",
        alpha=0.7,
        ax=axes[0]
    )
    axes[0].set_title(f"Histogram of Daily Standardized Residuals ({name})")
    axes[0].set_xlabel("Residual Value")
    axes[0].set_ylabel("Density")

    # Get the current x-axis limits from the histogram for plotting the t-distribution
    x_min, x_max = axes[0].get_xlim()
    x_vals = np.linspace(x_min, x_max, 200)
    # Compute the theoretical density for the t distribution with df=nu.
    # Note: the standardized residuals are assumed to be scaled appropriately.
    pdf_vals = stats.t.pdf(x_vals, df=nu)
    axes[0].plot(x_vals, pdf_vals, color="red", lw=2, label=f"t-dist (df={nu:.1f})")
    axes[0].legend()

    # ----- Right Subplot: Q-Q Plot against a t Distribution -----
    # Instead of using a normal distribution, we now compare against the t distribution with df=nu.
    stats.probplot(std_residuals[name], dist="t", sparams=(nu,), plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of Daily Standardized Residuals ({name})\nvs. t Distribution (df={nu:.1f})")

    plt.tight_layout()
    plt.show()



# Step 8 Plot ACF of Standardized Residuals & Squared Residuals (Daily Data)
# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None


# Determine the keys for S&P 500 and Dow Jones from std_residuals dictionary
sp_key = get_index_key(["S&P 500", "SP500", "S&P500"], std_residuals)
dj_key = get_index_key(["DowJones", "Dow Jones", "DJ"], std_residuals)

if sp_key is None:
    raise KeyError("S&P 500 key not found in std_residuals!")
if dj_key is None:
    raise KeyError("Dow Jones key not found in std_residuals!")

# Create a list with the two indices of interest
indices_to_plot = [sp_key, dj_key]

# Plot ACF for each index (for standardized residuals and their squares)
for name in indices_to_plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ACF of Daily Standardized Residuals
    sm.graphics.tsa.plot_acf(std_residuals[name].dropna(), lags=20, ax=axes[0])
    axes[0].set_title(f"ACF of Daily Standardized Residuals ({name})")
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Autocorrelation Coefficient")
    axes[0].text(0.5, -0.9, "Shaded region = 95% Confidence Interval",
                 transform=axes[0].transAxes, fontsize=10, color="gray", ha="center")

    # Plot ACF of Squared Daily Standardized Residuals
    sm.graphics.tsa.plot_acf(np.square(std_residuals[name].dropna()), lags=20, ax=axes[1])
    axes[1].set_title(f"ACF of Squared Daily Standardized Residuals ({name})")
    axes[1].set_xlabel("Lag (days)")
    axes[1].set_ylabel("Autocorrelation Coefficient")
    axes[1].text(0.5, -0.9, "Shaded region = 95% Confidence Interval",
                 transform=axes[1].transAxes, fontsize=10, color="gray", ha="center")

    plt.tight_layout()
    plt.show()


# Step 9: Ljung-Box Test (Daily Data)
from statsmodels.stats.diagnostic import acorr_ljungbox

# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None

# Determine the keys for S&P 500 and Dow Jones from std_residuals dictionary
sp_key = get_index_key(["S&P 500", "SP500", "S&P500"], std_residuals)
dj_key = get_index_key(["DowJones", "Dow Jones", "DJ"], std_residuals)

if sp_key is None:
    raise KeyError("S&P 500 key not found in std_residuals!")
if dj_key is None:
    raise KeyError("Dow Jones key not found in std_residuals!")

# Run Ljung-Box test for S&P 500 and Dow Jones
for name in [sp_key, dj_key]:
    res_clean = std_residuals[name].dropna()  # Clean daily residuals
    lb_test = acorr_ljungbox(res_clean, lags=[10], return_df=True)
    lb_test_sq = acorr_ljungbox(np.square(res_clean), lags=[10], return_df=True)

    print(f"\nLjung-Box Test Results for {name} - Daily Standardized Residuals:")
    print(lb_test)

    print(f"\nLjung-Box Test Results for {name} - Squared Daily Standardized Residuals:")
    print(lb_test_sq)


# Step 10
# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None

# Define assets and resolve the actual keys from stocks_data
assets = ["S&P 500", "DowJones"]

# Get correct keys for each asset
sp_key = "S&P 500"  # Assuming this key exists
dj_key = get_index_key(["DowJones", "Dow Jones", "DJ"], stocks_data)

if dj_key is None:
    raise KeyError("No appropriate key found for Dow Jones in stocks_data!")

asset_keys = {
    "S&P 500": sp_key,
    "DowJones": dj_key
}

for asset in assets:
    key = asset_keys[asset]
    print(f"\nProcessing for {asset} (using key: {key})...")

    # 1) Convert daily log returns to percentage terms
    returns_pct = process_data_daily(stocks_data[key]) * 100

    # Define forecast period: approximately one trading year (252 days)
    forecast_days = 252
    if len(returns_pct) < forecast_days + 21:
        forecast_days = len(returns_pct) - 21
        print(f"Adjusted forecast_days for {asset}: {forecast_days}")

    # 2) Compute the rolling one-step-ahead forecast (Predicted Volatility)
    rolling_vol_preds = []
    for i in range(forecast_days):
        # Use all available data up to the forecast date
        train = returns_pct[:-(forecast_days - i)]
        model = arch_model(train, vol="Garch", p=1, q=1, dist='t', rescale=True)
        fit = model.fit(disp="off")
        pred = fit.forecast(horizon=1)
        pred_vol = np.sqrt(pred.variance.values[-1, 0])
        rolling_vol_preds.append(pred_vol)
    predicted_vol_series = pd.Series(rolling_vol_preds, index=returns_pct.index[-forecast_days:])

    # 3) Compute Actual (Realized) Volatility using 21-day rolling window
    actual_vol_21 = returns_pct.rolling(window=21).std().dropna()[-forecast_days:]

    # 4) Plot the comparison in one figure
    plt.figure(figsize=(12, 5))
    plt.plot(predicted_vol_series, label="Predicted Vol (GARCH 1-step)", color="red", linewidth=2)
    plt.plot(actual_vol_21, label="Actual Vol (21-day Rolling)", color="blue", linewidth=2)
    plt.title(f"{asset} - Actual Volatility (21-day Rolling) vs. Predicted Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Step 11: Static Forecast of Volatility (Non-Rolling) with Actual Volatility Comparison
# Aligning the last N days so that the predicted and actual lines share the same date index

# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None


assets = ["S&P 500", "DowJones"]

for asset in assets:
    # Resolve the correct key in stocks_data for the asset
    if asset == "DowJones":
        key = get_index_key(["DowJones", "Dow Jones", "DJ"], stocks_data)
        if key is None:
            raise KeyError("No appropriate key found for Dow Jones in stocks_data!")
    else:
        key = asset

    print(f"\nProcessing static forecast for {asset} (using key: {key})...")


# Step 11: Static Forecast of Volatility (Non-Rolling) – Out-of-Sample with EWMA for Actual Volatility
# Step 11: Static Forecast of Volatility (Non-Rolling) – Out-of-Sample with Rolling Realized Volatility
# Aligning the forecasted values with the actual volatility computed on the forecast period

# Helper function to get the correct key from a dictionary given possible variants
def get_index_key(possible_names, dictionary):
    for name in possible_names:
        if name in dictionary:
            return name
    return None

assets = ["S&P 500", "DowJones"]

for asset in assets:
    # Resolve the correct key in stocks_data for the asset
    if asset == "DowJones":
        key = get_index_key(["DowJones", "Dow Jones", "DJ"], stocks_data)
        if key is None:
            raise KeyError("No appropriate key found for Dow Jones in stocks_data!")
    else:
        key = asset

    print(f"\nProcessing static forecast for {asset} (using key: {key})...")

    # 1) Convert daily log returns to percentage terms
    returns_pct = process_data_daily(stocks_data[key]) * 100

    # Define horizons (forecast periods)
    short_horizon = 5       # short-term: final 5 days out-of-sample forecast
    medium_horizon = 22     # medium-term: final 22 days out-of-sample forecast (~1 month)
    long_horizon = 252      # long-term: final 252 days out-of-sample forecast (~1 trading year)

    # --- Short-Horizon Forecast (Out-of-Sample) ---
    train_short = returns_pct[:-short_horizon]  # use all data except the last 5 days
    model_short = arch_model(train_short, vol="Garch", p=1, q=1, dist='t', rescale=True)
    fit_short = model_short.fit(disp="off")
    short_forecast = fit_short.forecast(horizon=short_horizon)
    short_volatility = np.sqrt(short_forecast.variance.values[-1, :])
    short_vol_series = pd.Series(short_volatility, index=returns_pct.index[-short_horizon:])

    # --- Medium-Horizon Forecast (Out-of-Sample) ---
    train_medium = returns_pct[:-medium_horizon]  # use all data except the last 22 days
    model_medium = arch_model(train_medium, vol="Garch", p=1, q=1, dist='t', rescale=True)
    fit_medium = model_medium.fit(disp="off")
    medium_forecast = fit_medium.forecast(horizon=medium_horizon)
    medium_volatility = np.sqrt(medium_forecast.variance.values[-1, :])
    medium_vol_series = pd.Series(medium_volatility, index=returns_pct.index[-medium_horizon:])

    # --- Long-Horizon Forecast (Out-of-Sample) ---
    train_long = returns_pct[:-long_horizon]  # use all data except the last 252 days
    model_long = arch_model(train_long, vol="Garch", p=1, q=1, dist='t', rescale=True)
    fit_long = model_long.fit(disp="off")
    long_forecast = fit_long.forecast(horizon=long_horizon)
    long_volatility = np.sqrt(long_forecast.variance.values[-1, :])
    long_vol_series = pd.Series(long_volatility, index=returns_pct.index[-long_horizon:])

    # 6) Compute Actual (Realized) Volatility using rolling windows matching the forecast horizon
    actual_short = returns_pct.rolling(window=short_horizon).std().dropna()[-short_horizon:]
    actual_medium = returns_pct.rolling(window=medium_horizon).std().dropna()[-medium_horizon:]
    actual_long = returns_pct.rolling(window=long_horizon).std().dropna()[-long_horizon:]

    # 7) Plotting: Create three subplots for Short-, Medium-, and Long-Term Forecasts
    plt.figure(figsize=(12, 15))

    # --- Short-Horizon Subplot ---
    plt.subplot(3, 1, 1)
    plt.plot(short_vol_series, marker='o', color='green', linewidth=2,
             label=f"Short-Horizon Forecast ({short_horizon} days)")
    plt.plot(actual_short, marker='x', color='blue', linewidth=2,
             label=f"Actual Volatility (Rolling {short_horizon} days)")
    plt.title(f"{asset}: Short-Horizon Out-of-Sample Forecast vs. Actual Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(True)

    # --- Medium-Horizon Subplot ---
    plt.subplot(3, 1, 2)
    plt.plot(medium_vol_series, marker='o', color='green', linewidth=2,
             label=f"Medium-Horizon Forecast ({medium_horizon} days)")
    plt.plot(actual_medium, marker='x', color='blue', linewidth=2,
             label=f"Actual Volatility (Rolling {medium_horizon} days)")
    plt.title(f"{asset}: Medium-Horizon Out-of-Sample Forecast vs. Actual Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(True)

    # --- Long-Horizon Subplot ---
    plt.subplot(3, 1, 3)
    plt.plot(long_vol_series, color='green', linewidth=2,
             label=f"Long-Horizon Forecast ({long_horizon} days)")
    plt.plot(actual_long, color='blue', linewidth=2,
             label=f"Actual Volatility (Rolling {long_horizon} days)")
    plt.title(f"{asset}: Long-Horizon Out-of-Sample Forecast vs. Actual Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


