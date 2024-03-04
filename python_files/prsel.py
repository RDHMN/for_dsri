import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Download stock data
symbols = ['PLD', 'NEE', 'SO', 'DIS', 'NFLX', 'JNJ', 'MDT', 'PG', 'KO', 'SBUX', 'MAR', 'HSBC', '^GSPC']
data = yf.download(symbols, start='2010-01-01', end='2020-12-31')['Adj Close']

# Compute log returns and drop NaNs
log_returns = np.log(data / data.shift(1)).dropna()

if len(log_returns) > (20 + 1):
    window_size = 50
    X, y = [], []
    for i in range(window_size, len(log_returns)):
        X.append(log_returns.iloc[i-window_size:i, :-1].values.flatten())
        y.append(log_returns.iloc[i, -1])

    X = np.array(X)
    y = np.array(y)

    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define the search space for hyperparameters
        space = {
            'units': hp.choice('units', [50, 100, 150]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
        }

        def train_model(params):
            model = Sequential([
                Dense(params['units'], activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(1)
            ])
            optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=2, verbose=0)
            loss = model.evaluate(X_test_scaled, y_test, verbose=0)
            return {'loss': loss, 'status': STATUS_OK}

        # Perform hyperparameter optimization
        trials = Trials()
        best = fmin(train_model, space, algo=tpe.suggest, max_evals=10, trials=trials)

        best_units = [50, 100, 150][best['units']]
        best_learning_rate = best['learning_rate']

        print("Best hyperparameters found:")
        print("Units:", best_units)
        print("Learning Rate:", best_learning_rate)

        # Rebuild the model with the best hyperparameters
        best_model = Sequential([
            Dense(best_units, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(1)
        ])
        best_optimizer = keras.optimizers.Adam(learning_rate=best_learning_rate)
        best_model.compile(optimizer=best_optimizer, loss='mean_squared_error')

        # Train the model with the best hyperparameters
        best_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=2, verbose=1)

        # Predict and plot
        predictions = best_model.predict(X_test_scaled).flatten()

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
