import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def plot_time_series(data, model_name='', order=1, forecast=None, title=""):
    """
    Plot a pandas Series representing a time series.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(1, len(data)+1)
    y = np.asarray(data)
    
    if forecast is None:
        ax.plot(x, y, label="Complete data", c="blue", alpha=0.5)
    else:
        forecast_horizon = len(forecast)
        x_forecast = x[-forecast_horizon:]
        ax.plot(x[:-forecast_horizon], y[:-forecast_horizon], label="Training data", c="blue", alpha=0.5)
        ax.plot(x[-forecast_horizon:], y[-forecast_horizon:], marker="o", label="Actual test data", c="green", alpha=0.5)
        ax.plot(x[-forecast_horizon:], forecast, marker="o", linestyle="--", label=f"{model_name}({order}) forecast", c="red", alpha=0.5)
        ax.axvline(x_forecast[0], linestyle=":", label="Forecast start")
    
    ax.set_title(title)
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_acf_results(data, lags=30):
    fig, ax = plt.subplots()
    plot_acf(data, lags=lags, ax=ax)
    plt.title("ACF (Autocorrelation Function)")
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel("", fontsize=14)
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    plt.tight_layout()
    plt.show()


def plot_pacf_results(data, lags=30):
    fig, ax = plt.subplots()
    plot_pacf(data, lags=lags, ax=ax)
    plt.title("PACF (Partial Autocorrelation Function)")
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel("", fontsize=14)
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    plt.tight_layout()
    plt.show()    


def residual_diagnostics(model, lags=20):

    residuals = np.asarray(model.resid)

    fig, ax = plt.subplots()
    
    # Plot residual ACF
    plot_acf(residuals, lags=lags, ax=ax, zero=False)
    plt.title("ACF of Residuals")

    # Perform Ljung-Box test (results are added in legend)
    df_lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)

    # Create manual legend handles LB test results
    mean_patch = mpatches.Patch(color='None', label=f"Residual Mean: {np.mean(residuals):.4f}")
    lb_stat_patch = mpatches.Patch(color='None', label=f"LB statistics: {list(df_lb['lb_stat'])[0]:.4f}")
    lb_pvalue_patch = mpatches.Patch(color='None', label=f"LB p-value: {list(df_lb['lb_pvalue'])[0]:.4f}")
    
    ax.legend(handles=[mean_patch, lb_pvalue_patch], loc='upper right', fontsize=12)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    plt.tight_layout()
    plt.show()