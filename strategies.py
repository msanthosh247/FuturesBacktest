import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt



def turtleTradeSignals(
    close_prices: np.ndarray,
    high_value: int = 20,
    low_value: int = 12,
    max_pyramiding: int = 4,
    allow_short : bool = True , 
) -> np.ndarray:
    """
    Calculate the position size based on the highest and lowest prices over a specified period.

    Args:
        close_prices (np.ndarray): Array of closing prices.
        high_value (int): Period for calculating the highest price.
        low_value (int): Period for calculating the lowest price.
        max_pyramiding (int): Maximum number of positions that can be pyramided.

    Returns:
        np.ndarray: Array of position sizes (1 for long, -1 for short, 0 for no position).
    """
    n = len(close_prices)
    position_size = np.zeros(n, dtype=int)  # Initialize position size array
    current_position = 0  # Track the current number of open positions

    for i in range(high_value, n - 1):
       
        highest_close_long = np.max(close_prices[i - high_value : i + 1])
        lowest_close_long = np.min(close_prices[i - low_value : i + 1])

        highest_close_short = np.max(close_prices[i - low_value : i + 1])
        lowest_close_short = np.min(close_prices[i - high_value : i + 1])

        long_condition = close_prices[i] == highest_close_long
        exit_long_condition = close_prices[i] == lowest_close_long and current_position > 0

        if allow_short :
            short_condition = close_prices[i] == lowest_close_short
            exit_short_condition = close_prices[i] == highest_close_short and current_position < 0
        else:
            short_condition = False
            exit_short_condition = False


        if long_condition and current_position < max_pyramiding:
            if current_position < 0:
                current_position = 0 
            current_position += 1
            position_size[i + 1 :] = current_position  

        elif exit_long_condition:
            current_position = 0
            position_size[i + 1 : ] = current_position  
            
        elif short_condition and current_position > -max_pyramiding:
            if current_position > 0 :
                current_position = 0
            current_position -= 1
            position_size[i + 1:] = current_position  

        elif exit_short_condition:
            current_position = 0
            position_size[i + 1:] = 0  # Exit short position
    return position_size


def plot_portfolio_growth(pf_values, dates):
    """
    Plots the portfolio growth over time with highlighted entry and exit windows, 
    and displays CAGR in the title.

    Parameters:
        pf_values (list or array): Portfolio values over time.
        dates (list or array): Corresponding dates for the portfolio values.
        entry_window (int): Index of the entry point in the data.
        exit_window_size (int): Number of periods after entry to consider as exit.

    Returns:
        None
    """

    # Calculate CAGR (Compound Annual Growth Rate)
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr = ((pf_values[-1] / pf_values[0]) ** (1 / years) - 1) * 100

    # Plot the portfolio value
    plt.figure(figsize=(14, 7))
    plt.plot(dates, pf_values, label="Portfolio Value", color="#1f77b4", linewidth=2)

    # Highlight entry and exit windows
   

    # Formatting the plot
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.title(f"Portfolio Growth Over Time (CAGR: {cagr:.2f}%)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()