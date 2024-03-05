# Collecting data, and reformatting to single csv file.
# Missing hour at 2023-03-26 02:00:00 due to daylight saving time
# Date is added and NaN's are added in place
# Finally NaN values are exchanged between forecasts and actuals

import pandas as pd
import os

# change working directory
path_folder = '/Users/raouldohmen/Documents/GitHub/Thesis/data/raw_data'
os.chdir(path_folder)


def get_fundamentals():
    # load data from files
    # nieuwe solar met latest forecast
    df1 = pd.read_csv('solar-renewables-elec.csv', usecols=['Date (CET)', 
                                                            'SOLAR OUTTURN', 
                                                            'SOLAR DA FORECAST (TENNET)',
                                                            'SOLAR FORECAST UNADJUSTED (ENAPPSYS)'])
    # coal en gas zelfde
    df2 = pd.read_csv('vsactual-byfuel-forecast.csv', usecols=['Date (CET)', 
                                                               'COAL', 
                                                               'COAL (FC)', 
                                                               'GAS', 
                                                               'GAS (FC)'])

    # zelfde voor demand, forecast blijft slecht
    df3 = pd.read_csv('history-forecast-demand.csv', usecols=['Date (CET)', 
                                                              'OUTTURN DEMAND', 
                                                              'D-1',
                                                              'TREND-ADJUSTED FORECAST'])

    # andere vars voor wind gebruiken met da/latest forecast
    df4 = pd.read_csv('wind-renewables-elec.csv', usecols=['Date (CET)', 
                                                           'WIND ONSHORE OUTTURN',
                                                           'WIND OFFSHORE OUTTURN', 
                                                           'WIND DA FORECAST (TENNET)',
                                                           'WIND FORECAST (TENNET)'])

    df5 = pd.read_csv('forecast-daprices-pricing.csv', usecols=['Date (CET)', 
                                                                'ACTUAL DA PRICE (EPEX)',
                                                                'MONTEL AI PRICE FORECAST'])

    df6 = pd.read_csv('daprices-pricing-elec.csv', usecols=['Date (CET)', 
                                                            'BELGIUM (BE)', 
                                                            'DENMARK (DK)',
                                                            'GREAT BRITAIN (GB)', 
                                                            'GERMANY (DE)', 
                                                            'NORWAY (NO)'])

    # rename vars
    df1.rename(columns={'SOLAR OUTTURN': 'solar'}, inplace=True)
    df1.rename(columns={'SOLAR DA FORECAST (TENNET)': 'solar_fc_da'}, inplace=True)
    df1.rename(columns={'SOLAR FORECAST UNADJUSTED (ENAPPSYS)': 'solar_fc_latest'}, inplace=True)

    df2.rename(columns={'COAL': 'coal'}, inplace=True)
    df2.rename(columns={'GAS': 'gas'}, inplace=True)
    df2.rename(columns={'COAL (FC)': 'coal_fc_da'}, inplace=True)
    df2.rename(columns={'GAS (FC)': 'gas_fc_da'}, inplace=True)

    df3.rename(columns={'OUTTURN DEMAND': 'demand'}, inplace=True)
    df3.rename(columns={'D-1': 'demand_fc_da'}, inplace=True)
    df3.rename(columns={'TREND-ADJUSTED FORECAST': 'demand_fc_latest'}, inplace=True)

    # merge on/offshore generation and drop individual vars
    df4 = df4.drop(df4.index[0])
    df4['wind'] = df4['WIND OFFSHORE OUTTURN'].astype(float) + df4['WIND ONSHORE OUTTURN'].astype(float)
    df4 = df4.drop(['WIND OFFSHORE OUTTURN', 'WIND ONSHORE OUTTURN'], axis=1)
    df4.rename(columns={'WIND DA FORECAST (TENNET)': 'wind_fc_da'}, inplace=True)
    df4.rename(columns={'WIND FORECAST (TENNET)': 'wind_fc_latest'}, inplace=True)

    df5.rename(columns={'ACTUAL DA PRICE (EPEX)': 'dayAhead_NL'}, inplace=True)
    df5.rename(columns={'MONTEL AI PRICE FORECAST': 'dayAhead_fc_NL'}, inplace=True)

    df6.rename(columns={'BELGIUM (BE)': 'dayAhead_BE'}, inplace=True)
    df6.rename(columns={'DENMARK (DK)': 'dayAhead_DK'}, inplace=True)
    df6.rename(columns={'GREAT BRITAIN (GB)': 'dayAhead_GB'}, inplace=True)
    df6.rename(columns={'GERMANY (DE)': 'dayAhead_DE'}, inplace=True)
    df6.rename(columns={'NORWAY (NO)': 'dayAhead_NO'}, inplace=True)

    # rename date cols
    df1.rename(columns={'Date (CET)': 'date'}, inplace=True)
    df2.rename(columns={'Date (CET)': 'date'}, inplace=True)
    df3.rename(columns={'Date (CET)': 'date'}, inplace=True)
    df4.rename(columns={'Date (CET)': 'date'}, inplace=True)
    df5.rename(columns={'Date (CET)': 'date'}, inplace=True)
    df6.rename(columns={'Date (CET)': 'date'}, inplace=True)

    # convert to dateTime format
    df1['date'] = pd.to_datetime(df1['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')
    df2['date'] = pd.to_datetime(df2['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')
    df3['date'] = pd.to_datetime(df3['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')
    df4['date'] = pd.to_datetime(df4['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')
    df5['date'] = pd.to_datetime(df5['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')
    df6['date'] = pd.to_datetime(df6['date'].str.replace('[', '').str.replace(']', ''), format='%d/%m/%Y %H:%M')

    # merge dataframes
    merged_df = pd.merge(df1, df2, on='date', how='outer')
    merged_df = pd.merge(merged_df, df3, on='date', how='outer')
    merged_df = pd.merge(merged_df, df5, on='date', how='outer')
    merged_df = pd.merge(merged_df, df6, on='date', how='outer')
    # drop first row
    merged_df = merged_df.drop(merged_df.index[0])
    merged_df = pd.merge(merged_df, df4, on='date', how='outer')
    

    # Extract hour and minute components
    merged_df['hour_of_day'] = merged_df['date'].dt.hour
    merged_df['minute_of_day'] = merged_df['date'].dt.minute

    # Create dummies for hour of day and minute of day
    hour_dummies = pd.get_dummies(merged_df['hour_of_day'], prefix='hour')
    minute_dummies = pd.get_dummies(merged_df['minute_of_day'], prefix='minute')

    return merged_df


def handle_nan(df):
    # merging creates a few duplicates
    new_df = df.drop_duplicates(subset=['date'], keep='first').set_index('date')

    # Create a complete datetime range that includes the missing date excluded due to DST
    start = new_df.index.min()
    end = new_df.index.max()
    complete_date_range = pd.date_range(start=start, end=end, freq='15min')

    # Reindex the DataFrame using the complete datetime range
    new_df = new_df.reindex(complete_date_range)
    # convert df to float
    new_df = new_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # check for missing values
    missing_values_summary = new_df.isnull().sum()
    # print(missing_values_summary)
    # print(40 * '-')

    # replace missing values of da_forecast and actuals with latest forecasts
    new_df.loc[new_df['solar'].isnull(), 'solar'] = new_df['solar_fc_latest']
    new_df.loc[new_df['solar_fc_da'].isnull(), 'solar_fc_da'] = new_df['solar_fc_latest']

    # replace da_forecast NaNs with actuals
    new_df.loc[new_df['coal_fc_da'].isnull(), 'coal_fc_da'] = new_df['coal']
    new_df.loc[new_df['gas_fc_da'].isnull(), 'gas_fc_da'] = new_df['gas']

    # replace da_forecast NaNs with latest fc
    new_df.loc[new_df['demand_fc_da'].isnull(), 'demand_fc_da'] = new_df['demand_fc_latest']

    # replace actuals and da_forecast NaNs with latest forecast//NaNs for latest fc with actuals
    new_df.loc[new_df['wind'].isnull(), 'wind'] = new_df['wind_fc_latest']
    new_df.loc[new_df['wind_fc_da'].isnull(), 'wind_fc_da'] = new_df['demand_fc_latest']
    new_df.loc[new_df['wind_fc_latest'].isnull(), 'wind_fc_latest'] = new_df['wind']

    # next we fill in values for the hour lost by DST (26/3/2023 02:00)
    extra_hour = pd.to_datetime('26/03/2023 02:00', format='%d/%m/%Y %H:%M')
    prev_hour = pd.to_datetime('26/03/2023 01:00', format='%d/%m/%Y %H:%M')

    # set solar to zero for extra hour
    new_df.at[extra_hour, 'solar'] = 0
    new_df.at[extra_hour, 'solar_fc_da'] = 0
    new_df.at[extra_hour, 'solar_fc_latest'] = 0

    # Set solar to zero for each quarter for the extra hour
    for i in range(0, 4):
        extra_quarter = extra_hour + pd.Timedelta(minutes=15 * i)
        new_df.at[extra_quarter, 'solar'] = 0

    # Set other variables to preceding value for the extra hour
    extra_hour_values = new_df.loc[prev_hour].values
    for quarter in range(1, 5):
        extra_quarter = extra_hour + pd.Timedelta(minutes=15 * (quarter - 1))
        new_df.loc[extra_quarter] = extra_hour_values

    # check if all NaNs are dealt with
    missing_values_summary = new_df.isnull().sum()
    print(missing_values_summary)
    # print(40 * '-')

    # add dateTime column
    new_df.insert(0, 'DateTime', new_df.index)

    return new_df


def get_leba():
    # Read the CSV file, correcting for any datetime parsing issues
    data = pd.read_csv('leba_prices.csv')

    # Attempting to directly remove timezone information during parsing
    data['From'] = pd.to_datetime(data['From'], errors='coerce', utc=True).dt.tz_convert(None)
    data['Until'] = pd.to_datetime(data['Until'], errors='coerce', utc=True).dt.tz_convert(None)

    # Prepare the date range for the entire year with 15-minute intervals
    date_range = pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:15", freq="15min")

    # Initialize an empty DataFrame with the desired date range
    df_final = pd.DataFrame(date_range, columns=['DateTime'])

    # For each row in the data, fill the corresponding values in the new DataFrame
    for index, row in data.iterrows():
        # Find the indexes in the date range that fall within the current row's period
        mask = (df_final['DateTime'] >= row['From']) & (df_final['DateTime'] < row['Until'])
        # Update the 'Money' column for these dates
        df_final.loc[mask, 'Money'] = row['Money']
    
    df_final.rename(columns={'Money': 'gas_price'}, inplace=True)

    return df_final


def get_midprice():
    # read csv file
    data = pd.read_csv('ladder_NL.csv')
    
    # Combine 'Date' and 'period_from' to create a DateTime column
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['period_from'])
    
    # Calculate the mid price as the average of 'minmin' and 'posmin'
    data['mid_price'] = (data['minmin'] + data['posmin']) / 2
    
    # Creating a new DataFrame with only DateTime and mid_price
    df_mid_price_only = data[['DateTime', 'mid_price']].set_index('DateTime')

    df_mid_price_only_dropped = df_mid_price_only[:-2]

    
    return df_mid_price_only


def save_data(df):
    # change working directory
    path_folder = '/Users/raouldohmen/Documents/GitHub/Thesis/data/clean_data'
    os.chdir(path_folder)
    # save merged df as 3 csv files: actuals, forecasts, and DA-prices
    actuals = df[['DateTime', 
                  'solar', 
                  'coal', 
                  'gas', 
                  'demand', 
                  'wind', 
                  'dayAhead_NL',
                  'gas_price', 
                  'mid_price']]
    forecasts = df[['DateTime', 
                    'solar_fc_da', 
                    'solar_fc_latest', 
                    'coal_fc_da',
                    'gas_fc_da', 
                    'demand_fc_da', 
                    'demand_fc_latest',
                    'wind_fc_da', 
                    'wind_fc_latest']]
    da_prices = df[['DateTime', 
                    'dayAhead_fc_NL', 
                    'dayAhead_NL', 
                    'dayAhead_BE',
                    'dayAhead_DK', 
                    'dayAhead_DE', 
                    'dayAhead_GB', 
                    'dayAhead_NO']]


    # save data to different csv files
    df.to_csv('dep_var.csv', index=False)
    actuals.to_csv('actuals.csv', index=False)
    forecasts.to_csv('forecasts.csv', index=False)
    da_prices.to_csv('da_prices.csv', index=False)

# read data
df_fund = get_fundamentals()
# handle NaNs
df_comp = handle_nan(df_fund)
# read and format LEBA data
df_leba = get_leba()
# read and format midprice
df_midprice = get_midprice()

# merge dataframes
df_comp = pd.merge(df_comp, df_leba, on='DateTime', how='outer')
df_comp = pd.merge(df_comp, df_midprice, on='DateTime', how='outer')

# drop last 2 rows as we only have daata until 23:15
df_comp = df_comp[:-2]
df_comp.set_index('DateTime', inplace=True)

#  next we fill in values for the hour lost by DST (26/3/2023 02:00) for midprice
extra_hour = pd.to_datetime('26/03/2023 02:00', format='%d/%m/%Y %H:%M')
prev_hour = pd.to_datetime('26/03/2023 01:00', format='%d/%m/%Y %H:%M')

extra_hour_values = df_comp.loc[prev_hour].values

for quarter in range(1, 5):
    extra_quarter = extra_hour + pd.Timedelta(minutes=15 * (quarter - 1))
    df_comp.loc[extra_quarter] = extra_hour_values

missing_values_summary = df_comp.isnull().sum()
print(missing_values_summary)


# save original and MinMax scaled dfs
# add dateTime column
df_comp.insert(0, 'DateTime', df_comp.index)
print(df_comp.shape)

save_data(df_comp)

