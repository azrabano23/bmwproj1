import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def preprocess_data(df):
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    columns = ['Close', '50_MA', '200_MA', 'Lag_1', 'Lag_2']
    scaled_data = scaler.fit_transform(df[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df, scaler

def train_linear_regression(scaled_df):
    X = scaled_df[['50_MA', '200_MA', 'Lag_1', 'Lag_2']]
    y = scaled_df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return lr_model, mse, predictions

def train_arima_model(df):
    arima_model = ARIMA(df['Close'], order=(5, 1, 0))
    arima_result = arima_model.fit()
    forecast = arima_result.forecast(steps=30)
    return arima_result, forecast

def build_and_train_lstm(scaled_df):
    X = np.array(scaled_df[['Lag_1', 'Lag_2']])
    y = np.array(scaled_df['Close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    predictions = lstm_model.predict(X_test)
    return lstm_model, predictions

def optimize_portfolio(df):
    returns = df['Close'].pct_change().dropna()
    mean_returns = mean_historical_return(df[['Close']])
    cov_matrix = CovarianceShrinkage(df[['Close']]).ledoit_wolf()
    ef = EfficientFrontier(mean_returns, cov_matrix)
    weights = ef.max_sharpe()
    perf = ef.portfolio_performance(verbose=True)
    return weights, perf

def trading_algorithm(predictions, df, initial_cash=10000):
    portfolio = {'cash': initial_cash, 'positions': 0}
    for day, price in enumerate(predictions):
        if price > df['Close'].iloc[day]:
            portfolio['positions'] += portfolio['cash'] / price
            portfolio['cash'] = 0
        elif price < df['Close'].iloc[day] and portfolio['positions'] > 0:
            portfolio['cash'] += portfolio['positions'] * price
            portfolio['positions'] = 0
    return portfolio
