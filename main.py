#import statements 
import pandas as pd 
import numpy as np 
import yfinance as yf
#neural network import statements
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Dataframe has monthly occuring data starting from 1991
#load dataset 
housing_df = pd.read_csv('HPI_master.csv')

#Data preprocessing 

#only filter for the country wide index
housing_df = housing_df[housing_df['level'].str.lower().str.contains('usa')]
housing_df = housing_df[housing_df['place_name'] == 'United States']
housing_df = housing_df[housing_df['frequency'] == 'monthly']

#drop season non-adjusted column 
housing_df = housing_df.drop('index_nsa', axis=1)

#convert the date to a datetime object 
housing_df['Date'] = pd.to_datetime(housing_df['yr'].astype(str) + '-' + housing_df['period'].astype(str), format='%Y-%m')

#filter for dates after 11/01/1998
housing_df = housing_df.loc[housing_df['Date'] >= '11/01/1998']

# Drop the original 'Year' and 'Month' columns if needed
final_housing_df = housing_df.drop(['yr', 'period'], axis=1)

#only select date and season adjusted housing index 
final_housing_df = housing_df[['Date', 'index_sa']]
final_housing_df['Date'] = pd.to_datetime(final_housing_df['Date'])
final_housing_df = final_housing_df.reset_index(drop=True)

#create an monthly inflation dataframe 

end_date = pd.to_datetime("2023-12-31")

# Generate date range from January 1998 to the present
date_range = pd.date_range(start="1998-01-01", end=end_date, freq="MS")

# Extract month-year combinations with numeric months
month_year_combinations = date_range.strftime('%m-%Y')

#inflation data 
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

#create the inflation Dataframe 
inflation_df = pd.DataFrame({'Date': month_year_combinations, 'Inflation': data_list})

#import VIX (volatility index) from yfinance API 
ticker_symbol = "^VIX"

#start and end dates for the data
start_date = "1998-11-30"
end_date = "2024-01-10"  #End date can be adjusted 

# Download the data
vix_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1wk")
vix_data = vix_data[['Open','Close']]
vix_data = vix_data.reset_index()

vix_data['Date'] = pd.to_datetime(vix_data['Date']) #convert date column to datetime object 

#rename columns for clarity 
vix_data = vix_data.rename(columns= {'Open': 'Vix Open','Close': 'Vix Close'})

# Merge Inflation and housing indedx dataFrames based on the 'Date' column
merged_df = pd.merge(inflation_df, final_housing_df, on='Date')


#import 10 year bond yield - weekly data starting from 1990 
ten_year_yield_df = pd.read_csv('10-year-yield.csv')

#data preprocessing 
yield_df = ten_year_yield_df.iloc[::-1]
yield_df = yield_df.reset_index(drop=True)
yield_df['Date'] = pd.to_datetime(yield_df['Date']) #date to datetime object 
yield_after1998_df = yield_df.loc[yield_df['Date'] >= '11/24/1998'] #filter dataframe for dates after 11/24/1998
new_yield_df = yield_after1998_df.reset_index(drop=True)

#reformat date column for consistency across each dataframe (for merge later)
new_yield_df['Date'] = new_yield_df['Date'].dt.strftime('%Y-%m-%d')

#import weekly SPY data from yfinance API 
spyticker = yf.Ticker("SPY")
SPY_df = spyticker.history(period="max", interval="1wk", start="1998-11-29", end="2024-01-03" , auto_adjust=True, rounding=True)

SPY_df = SPY_df.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1) #drop un-needed columns 
SPY_df.index = SPY_df.index.strftime('%Y-%m-%d') #reformat date column 
new_SPY = SPY_df.reset_index()

#merge bond yield and SPY data into one dataframe 
result = pd.merge(new_yield_df, new_SPY, left_index=True, right_index=True)
result = result.drop(['Open_x', 'High_x', 'Low_x', 'Date_x','High_y', 'Low_y'], axis=1) #drop redundant columns 

# Rename columns
result.rename(columns={'Date_y': 'Date', 'Change %' :'Bond % Change', 'Price':'Bond Price', 'Open_y': 'SPY Open', 'Close': 'SPY Close'}, inplace=True)

result['Date'] =  pd.to_datetime(result['Date']) #convert date to datetime object 

#merge result dataframe and vix data together 
result_df = pd.merge(vix_data, result, on='Date')

# Rearrange the order of columns
new_order = ['Date', 'Bond Price','Vix Open','Vix Close', 'Bond % Change', 'SPY Open', 'SPY Close', 'Volume']
result_df = result_df[new_order]

#create new dataframe to convert inflation and housing index from weekly format to monthly 
start_date = '1998-11-29'
end_date = pd.to_datetime('today')

# Generate a date range with weekly frequency
date_range = pd.date_range(start=start_date, end=end_date, freq='W-Mon')

# Create a DataFrame with the date range
df = pd.DataFrame({'Date': date_range})
df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = None
df['index_sa'] = None

#iterate through the dataframe and copy each monthly occurrence into weekly occurrences 
for index1, row1 in df.iterrows(): 
    for index2, row2 in merged_df.iterrows(): 
        if(row2['Date'].year == row1['Date'].year) & (row1['Date'].month == row2['Date'].month): 
            df.at[index1, 'Inflation'] = row2['Inflation']
            df.at[index1, 'index_sa'] = row2['index_sa']

#only take data from before 11/1/2023
filtered_df = df[df['Date'] <= '2023-11-01']
filtered_df['Date'] = pd.to_datetime(filtered_df['Date']) #convert date to datetime object 

#merge weekly inflation and housing data with the other result dataframe 
final_df = pd.merge(filtered_df, result_df, on='Date')

#create percent change columns for SPY and VIX indexes and SPY weekly volume
final_df['SPY % Change'] = ((final_df['SPY Open'] - final_df['SPY Close'])*100)/final_df['SPY Open'] 
final_df['VIX % Change'] = ((final_df['Vix Open'] - final_df['Vix Close'])*100)/final_df['Vix Open']
final_df['Volume % Change'] = final_df['Volume'].pct_change() * 100

final_df.fillna(0, inplace=True) #fill first row percent change with 0

#round columns to three decimal places 
columns_to_round = ['SPY % Change', 'Volume % Change', 'VIX % Change']
final_df[columns_to_round] = final_df[columns_to_round].round(3)

#reorder the columns 
final_order = ['Date', 'Inflation', 'index_sa', 'Bond % Change', 'SPY % Change', 'VIX % Change', 'Volume % Change', 'Bond Price','Volume', 'SPY Open','SPY Close', 'Vix Open', 'Vix Close']
final_df = final_df[final_order]

#reformat the bond % change column 
final_df['Bond % Change'] = final_df['Bond % Change'].str.replace('%', '')

# Convert the column to numeric type
final_df['Bond % Change'] = pd.to_numeric(final_df['Bond % Change'])

#final dataframe 
#print(final_df)

numerical_columns = ["Inflation", "index_sa",'Vix Close', "Bond Price", "SPY Open", "SPY Close", 'Volume']

#separate out features for prediction 
features = ['Inflation'	,'index_sa'	,'Bond % Change','VIX % Change','Volume % Change']
target = "SPY % Change"

# Split the data into training and testing sets
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

#create the neural network architecture 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_df[features], train_df[target], epochs=40, batch_size=8, validation_data=(test_df[features], test_df[target]))

# Evaluate the model on the test set
predictions = model.predict(test_df[features])
mse = mean_squared_error(test_df[target], predictions)
print(f"Mean Squared Error on the test set: {mse}")