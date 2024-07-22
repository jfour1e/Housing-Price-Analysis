### The goal of this project is to take various macroeconomic indicators and attempt to predict future returns of the stock market (using the s&p500 Index). 

The features used in this project were: 
- Inflation data
- HPI (housing price indicator) seasonally adjusted
- VIX (volatility Index)
- Bond prices

Data preprocesing:
------------------
All data was normalized (logarithmically) and lined up in a dataframe according to timestamp
Data occurs in monthly frequency due to inflation and HPI indicator being updated monthly. 

Conclusion
----------
Ultimately it was found that inflation is highly correlated with S&P500 open and close prices (correlation coefficienct of 0.65. Additionally, the HPI indicator was correlated with S&P500 open and close prices (correlation coefficient of 0.93) 

