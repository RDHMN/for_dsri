import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


# Download stock data
# symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'KO', 'HSBC', '^GSPC']  # 5 stocks + S&P 500 (^GSPC)
symbols = ['PLD', 'NEE', 'SO', 'DIS', 'NFLX', 'JNJ', 'MDT', 'PG', 'KO', 'SBUX', 'MAR', 'HSBC', '^GSPC']  # 5 stocks + S&P 500 (^GSPC)
data = yf.download(symbols, start='2010-01-01', end='2020-12-31')['Adj Close']

print(data.shape)

# Compute log returns and drop NaNs
log_returns = 100
log_returns += 100 * np.log(data / data.shift(1)).dropna()

# Ensure there's enough data after dropping NaNs
if len(log_returns) > (20 + 1):  # Window size + 1
    window_size = 50
    X, y = [], []
    for i in range(window_size, len(log_returns)):
        X.append(log_returns.iloc[i-window_size:i, :-1].values.flatten())  # Stocks' log returns
        y.append(log_returns.iloc[i, -1])  # S&P 500 log return

    X = np.array(X)
    y = np.array(y)

    # Check if we have enough samples for the split
    if len(X) > 10:  # Arbitrary minimum number to allow for a test split
        # Split and scale the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and compile the MLP model
        model = Sequential([
            Dense(100, activation='linear', input_shape=(X_train_scaled.shape[1],)),
            Dense(60, activation='linear'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.00528183360234428)
        model.compile(optimizer=optimizer, loss='mse')

        # lr = 0.00528183360234428

        # Train the model
        model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=2, verbose=1)

        # loaded_model = load_model('mlp_test.h5')

        # Predict and plot
        predictions = model.predict(X_test_scaled).flatten()

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual S&P 500 Log Returns')
        plt.plot(predictions, label='Predicted S&P 500 Log Returns')
        plt.title('S&P 500 Log Return Prediction using MLP')
        plt.xlabel('Time')
        plt.ylabel('Log Return')
        plt.legend()
        plt.show()
    else:
        print("Not enough data for splitting after preprocessing.")
else:
    print("Not enough data after dropping NaNs.")