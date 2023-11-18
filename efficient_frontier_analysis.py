#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: robertisaksen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from RP_and_OP import return_portfolios, optimal_portfolio

# 1. Load the stock data
stock_data = pd.read_csv('stock_data_weak.csv')

# Display the loaded data
print(stock_data)

# 2. Find the quarterly returns over each period for all assets
returns_quarterly = stock_data.pct_change()

# 3. Find the expected return for each asset
expected_returns = returns_quarterly.mean()

# 4. Find the covariance of the quarterly returns over each period
cov_quarterly = returns_quarterly.cov()

# Calculate the standard deviation of each single asset
single_asset_std = np.sqrt(np.diagonal(cov_quarterly))

# 5. Use the expected returns and covariances to find a set of random portfolios
random_portfolios = return_portfolios(expected_returns, cov_quarterly)

# 6. Plot the set of random portfolios
plt.scatter(random_portfolios['Volatility'], random_portfolios['Returns'], marker='o', color='blue', alpha=0.1)
plt.ylabel('Expected Returns', fontsize=14)
plt.xlabel('Volatility (Std. Deviation)', fontsize=14)
plt.title('Random Portfolios', fontsize=24)
plt.show()

# 7. Use the optimal_portfolio() function to calculate the efficient frontier
weights, returns, risks = optimal_portfolio(returns_quarterly)

# 8. Plot the efficient frontier
plt.scatter(single_asset_std, expected_returns, marker='X', color='red', s=200)
plt.plot(risks, returns, color='yellow', marker='o', linestyle='dashed', linewidth=2, markersize=10)
plt.ylabel('Expected Returns', fontsize=14)
plt.xlabel('Volatility (Std. Deviation)', fontsize=14)
plt.title('Efficient Frontier', fontsize=24)
plt.show()
