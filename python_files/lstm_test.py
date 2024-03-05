import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Download stock data
# symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', '^GSPC']  # 5 stocks + S&P 500 (^GSPC)
symbols = ['PLD', 'NEE', 'SO', 'DIS', 'NFLX', 'JNJ', 'MDT', 'PG', 'KO', 'SBUX', 'MAR', 'HSBC', '^GSPC']  # 5 stocks + S&P 500 (^GSPC)
data = yf.download(symbols, start='2010-01-01', end='2020-12-31')['Adj Close']

# Compute log returns and drop NaNs
log_returns = np.log(data / data.shift(1)).dropna()

# Prepare the data for LSTM
window_size = 20
X, y = [], []
for i in range(window_size, len(log_returns)):
    X.append(log_returns.iloc[i-window_size:i, :-1].values)  # Keeping the shape for LSTM
    y.append(log_returns.iloc[i, -1])  # S&P 500 log return

X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the data with RobustScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
# Since RobustScaler does not support 3D data directly, we reshape the data for scaling, then reshape it back
X_train_scaled = np.array([scaler.fit_transform(x) for x in X_train])
X_test_scaled = np.array([scaler.transform(x) for x in X_test])

# Build and compile the LSTM model with two LSTM layers
model = Sequential([
    # LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, X_train_scaled.shape[2])),
    LSTM(50, activation='relu', input_shape=(window_size, X_train_scaled.shape[2])),
    # LSTM(15, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Predict and plot
predictions = model.predict(X_test_scaled).flatten()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual S&P 500 Log Returns')
plt.plot(predictions, label='Predicted S&P 500 Log Returns')
plt.title('S&P 500 Log Return Prediction using Double LSTM Layers')
plt.xlabel('Time')
plt.ylabel('Log Return')
plt.legend()
plt.show()