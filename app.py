# Import Libraries

import numpy as np 
import pandas as pd 
import yfinance as yf
from keras.model import load_model
import streamlit as st

# Load Model

model = load_model('/Users/shayan/Work/StockPricePrediction/Stock-Market-Prediction/Stock-Market-Prediction/Stock Price Prediction Model.keras')

st.header('Stock Market Predictor')


# Load Data and Symbols

stock = st.text_input('Enter Stock Symbol', 'BTC-USD')
start = '2014-01-01'
end = '2024-01-01'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Choose and fix train and test relativity 

data_train = data.Close[0: int(len(data)*0.80)]
data_test = data.Close[int(len(data)*0.80): len(data)]

# Scaler

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0,1))


# Convert numpy array to pandas DataFrame or Series for test data

data_train_df = pd.DataFrame(data_train)
data_test_df = pd.DataFrame(data_test)

past_100_days = data_train_df.tail(100)

data_test = pd.concat([past_100_days, data_test_df], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test_df)



x = []
y = []

for i in range (100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x = np.array(x)
y = np.array(y)