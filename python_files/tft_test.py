# Problem with pytorch forecasting library

import yfinance as yf
import pandas as pd
import numpy as np
import pytorch_forecasting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# Download stock data
symbols = ['PLD', 'NEE', 'SO', 'DIS', 'NFLX', 'JNJ', 'MDT', 'PG', 'KO', 'SBUX', 'MAR', 'HSBC', '^GSPC']
data = yf.download(symbols, start='2010-01-01', end='2020-12-31')['Adj Close']

# Compute log returns and drop NaNs
log_returns = np.log(data / data.shift(1)).dropna()
log_returns.reset_index(inplace=True)

# Preparing data for TFT
log_returns['time_idx'] = log_returns.index
log_returns['series'] = 1  # Assuming all data belongs to one series

# Split the data
training_cutoff = log_returns['time_idx'].max() - int(len(log_returns) * 0.1)
train_df = log_returns[log_returns['time_idx'] <= training_cutoff]
val_df = log_returns[log_returns['time_idx'] > training_cutoff]

# Create the dataset for PyTorch Forecasting
max_encoder_length = 60
max_prediction_length = 20

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="^GSPC",
    group_ids=["series"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["series"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[symbol for symbol in symbols if symbol != "^GSPC"] + ["^GSPC"],
    target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

batch_size = 128
train_dataloader = DataLoader(training, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation, batch_size=batch_size * 10, shuffle=False)

# Define the Temporal Fusion Transformer model
pl_trainer_kwargs = {"gpus": 0, "gradient_clip_val": 0.1}
trainer = pl.Trainer(
    max_epochs=10,
    gpus=0,
    gradient_clip_val=0.1,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Train the model
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Assuming `trainer` is your PyTorch Lightning trainer and `tft` is your model
# Get predictions
raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)

# Convert raw predictions to actual values
# Note: Adjust the following lines according to your data and prediction details
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = raw_predictions["prediction"].mean(1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actuals[:100], label='Actuals')
plt.plot(predictions[:100], label='Predictions')
plt.title('TFT Predictions vs Actuals')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Interpret model output
interpretation = tft.interpret_output(raw_predictions, reduction="sum")
feature_importance = interpretation["attention"]

# Summarize feature importance
feature_importance_sum = feature_importance.sum(dim=1).mean(dim=0)
features = training.time_varying_unknown_reals + training.time_varying_known_reals
importance_df = pd.DataFrame(feature_importance_sum.cpu().numpy(), index=features, columns=["Importance"])

# Sort and plot
importance_df = importance_df.sort_values(by="Importance", ascending=False)
importance_df.plot(kind='bar', figsize=(10, 6), title="Feature Importance")
plt.show()
