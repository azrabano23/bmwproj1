# bmwproj1

# What the Program Does:

Data Loading and Preprocessing:

Loads historical stock or price data.
Creates rolling averages (50-day and 200-day moving averages) and lag features to capture historical trends.
Scales the data to enhance machine learning model performance.
Forecasting:

Linear Regression: Predicts close prices based on historical data and technical indicators.
ARIMA: Forecasts time series data using statistical properties of the dataset.
LSTM: Implements a neural network for advanced time series forecasting.
Portfolio Optimization:

Calculates mean returns and covariance matrices of stock returns.
Uses the Efficient Frontier to allocate resources optimally, maximizing risk-adjusted returns.
Algorithmic Trading Simulator:

Simulates buy-and-sell decisions based on predicted prices.
Tracks portfolio growth over time based on an initial cash investment.
Backtesting and Performance Analysis:

Measures strategy performance using metrics like Mean Squared Error (MSE) for regression models.
Outputs portfolio performance, including Sharpe ratio and volatility.


# Technical Skills Used:

Programming Language:

Python: The primary language for implementing machine learning models, statistical analysis, and trading simulations.
Machine Learning & Deep Learning:

Scikit-Learn: For linear regression and model evaluation.
TensorFlow/Keras: To build and train LSTM models for time series forecasting.
Statsmodels: For implementing ARIMA, a statistical forecasting technique.
Data Manipulation & Analysis:

Pandas: For dataset loading, preprocessing, and manipulation.
NumPy: For efficient numerical computations.
Data Visualization:

Libraries like Matplotlib or Seaborn can be integrated (though not explicitly included in your program) to visualize results such as forecasts and backtests.
Financial Modeling:

PyPortfolioOpt: For portfolio optimization using risk-return metrics like the Sharpe ratio.
Statistical calculations for returns and covariances to inform efficient frontier modeling.
Development Tools:

Google Colab: Ideal for training models with GPU acceleration.
VS Code: A versatile IDE for writing, debugging, and running the program locally.

# Libraries for Scaling:

Scikit-Learn's MinMaxScaler: Used to normalize data for better performance with machine learning models.

Time Series Analysis: Implementation of rolling averages and lag features for feature engineering.

ARIMA modeling for forecasting trends in close prices.
