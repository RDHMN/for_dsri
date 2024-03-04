'''
Exploratory data analysis of dependent variables created in data preprocessing file
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import seaborn as sns
import os

# change working directory
path_folder = '/Users/raouldohmen/Documents/GitHub/Thesis/data/clean_data'
os.chdir(path_folder)

# df = pd.read_csv('actuals.csv')
df = pd.read_csv('midprice_dataset.csv')
print(df.tail)


def plot_individual_vars(df):
    for column in df.columns:
        if column != 'DateTime':  # Exclude the datetime column by name
            plt.figure(figsize=(10, 4))
            plt.plot(df[column])
            plt.title(column)
            plt.grid()
            plt.ylabel('Value')
            plt.xlabel('Index')
            plt.show()


def plot_individual_hist(df):
    for column in df.columns:
        if column != 'DateTime':  # Exclude the datetime column by name
            plt.figure(figsize=(10, 4))
            df[column].hist(bins=20)
            plt.title(f'Histogram of {column}')
            plt.ylabel('Frequency')
            plt.xlabel('Value')
            plt.show()


def sum_stats(df):
    for column in df.columns:
        if column != 'DateTime':  # Exclude the datetime column by name
            print(f"Summary statistics for {column}:")
            print(df[column].describe())
            print("\n" + "-"*40 + "\n")


def corr_matrix(df):
    # Drop the datetime column
    # df_without_datetime = df.drop(columns=['X_merge'])
    
    # Compute the correlation matrix for the remaining columns
    # corr_matrix = df_without_datetime.corr()
    corr_matrix = df.corr()
    print(corr_matrix)
    
    # Print the correlation matrix
    # print("Correlation Matrix:")
    # print(corr_matrix)
    
    # Optional: Plot the correlation matrix for a visual representation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


    # Exclude non-numeric columns before calculating the correlation matrix
    all_cols = ['solar', 'solar_fc_da', 'solar_fc_latest',
                'coal', 'coal_fc_da', 'gas', 'gas_fc_da',
                'demand', 'demand_fc_da', 'demand_fc_latest',
                'wind', 'wind_fc_da', 'wind_fc_latest', 'dayAhead_NL', 'dayAhead_fc_NL']
    df_all = df[all_cols]
    corr_mat1 = df_all.corr()
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat1, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()

    actuals_only = ['solar', 'coal', 'dayAhead_NL', 'gas', 'demand', 'wind']
    df_actuals = df[actuals_only]
    corr_mat2 = df_actuals.corr()
    # Plot heatmap
    sns.heatmap(corr_mat2, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()

    forecast_only = ['dayAhead_NL', 'dayAhead_BE',
                     'dayAhead_DK', 'dayAhead_DE', 'dayAhead_GB', 'dayAhead_NO']
    df_forecast = df[forecast_only]
    corr_mat3 = df_forecast.corr()
    # Plot heatmap
    sns.heatmap(corr_mat3, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()


def calc_error(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    # Calculate errors
    mae = np.mean(np.abs(actual - forecast))
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)

    # Conditionally calculate MAPE
    mape = None  # Default to None if there are zero values in 'actual'
    if not np.any(actual == 0):
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    return {
        'MAE': np.round(mae, decimals=4),
        'MSE': np.round(mse, decimals=4),
        'RMSE': np.round(rmse, decimals=4),
        'MAPE': mape  # This will be None if 'actual' contains zeros
    }


def mape_zeros(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    # Filter out positions where actual is zero
    non_zero_positions = actual != 0
    actual_filtered = actual[non_zero_positions]
    forecast_filtered = forecast[non_zero_positions]

    # Calculate MAPE over non-zero actual values
    mape = np.mean(np.abs((actual_filtered - forecast_filtered) / actual_filtered)) * 100

    return mape


def hist_da_prices():
    plt.hist(df['dayAhead_DE'], bins=500, color='blue')
    plt.hist(df['dayAhead_NL'], bins=50, color='orange')
    plt.hist(df['dayAhead_BE'], bins=50, color='yellow')
    plt.hist(df['dayAhead_NO'], bins=50, color='red')
    plt.grid()
    plt.show()


if __name__ == "__main__":

    # plot_individual_vars(df)
    # plot_individual_hist(df)
    # sum_stats(df)
    corr_matrix(df)





    # specify columns to plot
    # groups = [0, 1, 2, 3, 5, 6, 7]
    # i = 1
    # # plot each column
    # pyplot.figure()
    # for group in groups:
    #     pyplot.subplot(len(groups), 1, i)
    #     pyplot.plot(df[:, group])
    #     pyplot.title(df.columns[group], y=0.5, loc='right')
    #     i += 1
    # pyplot.show()









    '''
    print('Solar: ', calc_error(df['solar'], df['solar_fc_da']))
    print('Gas: ', calc_error(df['gas'], df['gas_fc_da']))
    print('Coal : ', calc_error(df['coal'], df['coal_fc_da']))
    print('Demand: ', calc_error(df['demand'], df['demand_fc_da']))
    print('Wind: ', calc_error(df['wind'], df['wind_fc_da']))
    print('DA-price: ', calc_error(df['dayAhead_NL'], df['dayAhead_fc_NL']))
    print(40 * '-')

    print('Solar MAPE ignoring zeros: ', mape_zeros(df['solar'], df['solar_fc_da']))
    print('Coal MAPE ignoring zeros: ', mape_zeros(df['coal'], df['coal_fc_da']))
    print('DA-price MAPE ignoring zeros: ', mape_zeros(df['dayAhead_NL'], df['dayAhead_fc_NL']))
    coal and da-price MAPE inflated due to values close to zero
    '''
    