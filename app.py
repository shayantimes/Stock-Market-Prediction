# Import Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
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


# ploting MA's and Prices
st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)


st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)


x = []
y = []

for i in range (100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x = np.array(x)
y = np.array(y)


predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale

y = y * scale

# Plotting Price and Prediction
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)