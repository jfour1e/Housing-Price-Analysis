import pandas as pd 
import numpy as np 
import yfinance as yf

#monthly - from 1991
housing_df = pd.read_csv('HPI_master.csv')
housing_df = housing_df[housing_df['level'].str.lower().str.contains('usa')]
housing_df = housing_df[housing_df['place_name'] == 'United States']
housing_df = housing_df[housing_df['frequency'] == 'monthly']
housing_df = housing_df.drop('index_nsa', axis=1)
housing_df = housing_df.reset_index(drop=True)
housing_df['Date'] = pd.to_datetime(housing_df['yr'].astype(str) + '-' + housing_df['period'].astype(str), format='%Y-%m')
housing_df = housing_df.loc[housing_df['Date'] >= '11/01/1998']

# Drop the original 'Year' and 'Month' columns if needed
final_housing_df = housing_df.drop(['yr', 'period'], axis=1)
final_housing_df = housing_df[['Date', 'index_sa']]
final_housing_df = final_housing_df.reset_index(drop=True)
final_housing_df['Date'] = pd.to_datetime(final_housing_df['Date'])


# Assuming the current date is 2024-01-03
end_date = pd.to_datetime("2023-12-31")

# Generate the date range from January 1998 to the present
date_range = pd.date_range(start="1998-01-01", end=end_date, freq="MS")

# Extract month-year combinations with numeric months
month_year_combinations = date_range.strftime('%m-%Y')

data_list = [
    2.2, 2.3, 2.1, 2.1, 2.2, 2.2, 2.2, 2.5, 2.5, 2.3, 2.3, 2.4, 
    2.4, 2.1, 2.1, 2.2, 2.0, 2.1, 2.1, 1.9, 2.0, 2.1, 2.1, 1.9, 
    2.0, 2.2, 2.4, 2.3, 2.4, 2.5, 2.5, 2.6, 2.6, 2.5, 2.6, 2.6, 
    2.6, 2.7, 2.7, 2.6, 2.5, 2.7, 2.7, 2.7, 2.6, 2.6, 2.8, 2.7, 
    2.6, 2.6, 2.4, 2.5, 2.5, 2.3, 2.2, 2.4, 2.2, 2.2, 2.0, 1.9, 
    1.9, 1.7, 1.7, 1.5, 1.6, 1.5, 1.5, 1.3, 1.2, 1.3, 1.1, 1.1, 
    1.1, 1.2, 1.6, 1.8, 1.7, 1.9, 1.8, 1.7, 2.0, 2.0, 2.2, 2.2,
    2.3, 2.4, 2.3, 2.2, 2.2, 2.0, 2.1, 2.1, 2.0, 2.1, 2.1, 2.2,
    2.1, 2.1, 2.1, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 2.7, 2.6, 2.6,
    2.7, 2.7, 2.5, 2.3, 2.2, 2.2, 2.2, 2.1, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.3, 2.4, 2.3, 2.3, 2.4, 2.5, 2.5, 2.5, 2.2, 2.0, 1.8,
    1.7, 1.8, 1.8, 1.9, 1.8, 1.7, 1.5, 1.4, 1.5, 1.7, 1.7, 1.8,
    1.6, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.8, 0.8,
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.0, 2.1, 2.2, 2.2,
    2.3, 2.2, 2.3, 2.3, 2.3, 2.2, 2.1, 1.9, 2.0, 2.0, 1.9, 1.9,
    1.9, 2.0, 1.9, 1.7, 1.7, 1.6, 1.7, 1.8, 1.7, 1.7, 1.7, 1.7,
    1.6, 1.6, 1.7, 1.8, 2.0, 1.9, 1.9, 1.7, 1.7, 1.8, 1.7, 1.6,
    1.6, 1.7, 1.8, 1.8, 1.7, 1.8, 1.8, 1.8, 1.9, 1.9, 2.0, 2.1,
    2.2, 2.3, 2.2, 2.1, 2.2, 2.2, 2.2, 2.3, 2.2, 2.1, 2.1, 2.2,
    2.3, 2.2, 2.0, 1.9, 1.7, 1.7, 1.7, 1.7, 1.7, 1.8, 1.7, 1.8,
    1.8, 1.8, 2.1, 2.1, 2.2, 2.3, 2.4, 2.2, 2.2, 2.1, 2.2, 2.2,
    2.2, 2.1, 2.0, 2.1, 2.0, 2.1, 2.2, 2.4, 2.4, 2.3, 2.3, 2.3,
    2.3, 2.4, 2.1, 1.4, 1.2, 1.2, 1.6, 1.7, 1.7, 1.6, 1.6, 1.6,
    1.4, 1.3, 1.6, 3.0, 3.8, 4.5, 4.3, 4.0, 4.0, 4.6, 4.9, 5.5,
    6.0, 6.4, 6.5, 6.2, 6.0, 5.9, 5.9, 6.3, 6.6, 6.3, 6.0, 5.7,
    5.6, 5.5, 5.6, 5.5, 5.3, 4.8, 4.7, 4.3, 4.1, 4.0, 4.0, 0.0
]

inflation_df = pd.DataFrame({'Date': month_year_combinations, 'Inflation': data_list})
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'])

# Merge DataFrames based on the 'Date' column
merged_df = pd.merge(inflation_df, final_housing_df, on='Date')

#weekly from 1990 
ten_year_yield_df = pd.read_csv('10-year-yield.csv')
yield_df = ten_year_yield_df.iloc[::-1]
yield_df = yield_df.reset_index(drop=True)
yield_df['Date'] = pd.to_datetime(yield_df['Date'])
yield_after1998_df = yield_df.loc[yield_df['Date'] >= '11/24/1998']
new_yield_df = yield_after1998_df.reset_index(drop=True)

new_yield_df['Date'] = new_yield_df['Date'].dt.strftime('%Y-%m-%d')


spyticker = yf.Ticker("SPY")
SPY_df = spyticker.history(period="max", interval="1wk", start="1998-11-29", end="2024-01-03" , auto_adjust=True, rounding=True)
SPY_df = SPY_df.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1)
SPY_df.index = SPY_df.index.strftime('%Y-%m-%d')
new_SPY = SPY_df.reset_index()

result = pd.merge(new_yield_df, new_SPY, left_index=True, right_index=True)
result = result.drop(['Open_x', 'High_x', 'Low_x', 'Date_x','High_y', 'Low_y', 'Volume' ], axis=1)

# Rename columns
result.rename(columns={'Date_y': 'Date', 'Open_y': 'SPY Open', 'Close': 'SPY Close'}, inplace=True)
new_order = ['Date', 'Price', 'Change %', 'SPY Open', 'SPY Close']

# Rearrange the order of columns
result = result[new_order]

start_date = '1998-11-29'
end_date = pd.to_datetime('today')

# Generate a date range with weekly frequency
date_range = pd.date_range(start=start_date, end=end_date, freq='W-Mon')

# Create a DataFrame with the date range
df = pd.DataFrame({'Date': date_range})
df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = None
df['index_sa'] = None


for index1, row1 in df.iterrows(): 
    for index2, row2 in merged_df.iterrows(): 
        if(row2['Date'].year == row1['Date'].year) & (row1['Date'].month == row2['Date'].month): 
            df.at[index1, 'Inflation'] = row2['Inflation']
            df.at[index1, 'index_sa'] = row2['index_sa']

filtered_df = df[df['Date'] <= '2023-11-01']
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
result['Date'] = pd.to_datetime(result['Date'])

final_df = pd.merge(filtered_df, result, on='Date')

