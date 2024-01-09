import pandas as pd 
import numpy as np 
import yfinance as yf

#monthly - from 1991
housing_df = pd.read_csv('HPI_master.csv')
housing_df = housing_df[housing_df['level'].str.lower().str.contains('usa')]
housing_df = housing_df[housing_df['place_name'] == 'United States']
housing_df = housing_df[housing_df['frequency'] == 'monthly']
housing_df = housing_df.drop('NSA adjusted', axis=1)
housing_df = housing_df.reset_index(drop=True)

#yearly
data = {
    'Year': [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990, 1989, 1988, 1987, 1986, 1985, 1984, 1983, 1982, 1981, 1980, 1979, 1978, 1977, 1976, 1975, 1974, 1973, 1972, 1971, 1970, 1969, 1968, 1967, 1966, 1965, 1964, 1963, 1962, 1961, 1960],
    'Inflation Rate (%)': [8.00, 4.70, 1.23, 1.81, 2.44, 2.13, 1.26, 0.12, 1.62, 1.46, 2.07, 3.16, 1.64, -0.36, 3.84, 2.85, 3.23, 3.39, 2.68, 2.27, 1.59, 2.83, 3.38, 2.19, 1.55, 2.34, 2.93, 2.81, 2.61, 2.95, 3.03, 4.24, 5.40, 4.83, 4.08, 3.66, 1.90, 3.55, 4.30, 3.21, 6.13, 10.33, 13.55, 11.25, 7.63, 6.50, 5.74, 9.14, 11.05, 6.18, 3.27, 4.29, 5.84, 5.46, 4.27, 2.77, 3.02, 1.59, 1.28, 1.24, 1.20, 1.07, 1.46],
    'Annual Change': [3.30, 3.46, -0.58, -0.63, 0.31, 0.87, 1.14, -1.50, 0.16, -0.60, -1.09, 1.52, 2.00, -4.19, 0.99, -0.37, -0.17, 0.72, 0.41, 0.68, -1.24, -0.55, 1.19, 0.64, -0.79, -0.59, 0.13, 0.20, -0.34, -0.08, -1.21, -1.16, 0.57, 0.75, 0.41, 1.77, -1.65, -0.75, 1.09, -2.92, -4.20, -3.21, 2.29, 3.62, 1.13, 0.76, -3.40, -1.91, 4.88, 2.91, -1.02, -1.55, 0.38, 1.19, 1.50, -0.24, 1.43, 0.31, 0.04, 0.04, 0.13, -0.39, -0.39]
}
inflation_df = pd.DataFrame(data)

#weekly
ten_year_yield_df = pd.read_csv('10-year-yield.csv')

spyticker = yf.Ticker("SPY")
SPY_df = spyticker.history(period="max", interval="1wk", start="1998-12-01", end="2023-01-01" , auto_adjust=True, rounding=True)
SPY_df = SPY_df.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1)
SPY_df.head()
#only important column is spy close 
#data from 1998