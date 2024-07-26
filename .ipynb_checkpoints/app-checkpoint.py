import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt


model = load_model('E:\Cryptocurrency price prediction\Cryptocurrency price Predictions Model.keras')


st.markdown('# Cryptocurrency Price Prediction')
st.image("1590412750064.jpeg")
st.markdown('Crypto Price Prediction is the webapp based on Machine Learning model that predicts the future closing price.')

st.markdown('----')
st.markdown('# Language/Libraries used?')
st.markdown('API/ Web Scraper: CryptoCmd python scraper scraped the data from CoinMarketCap')
st.markdown('Data Wrangling: Pandas, Numpy')
st.markdown('Data Visualisation: Matplotlib, Seaborn')
st.markdown('Data Modeling: Scikit: Learn, TensorFlow, Keras')

st.markdown('Webapp: Streamlit')
st.markdown('IDE: Jupyter Notebook/ PyCharm')

st.markdown('----')
st.markdown('# Which Machine Learning Model is used in this webapp?')
st.markdown('LSTM is used in this webapp.')
st.markdown('LSTMs are widely used for sequence prediction problems and have proven to be extremely effective. The reason they work so well is that LSTM is able to store past information that is important and forget the information that is not.')
st.markdown('LSTM has three gates:')
st.markdown('a. The input gate: The input gate adds information to the cell state.')
st.markdown('b. The forget gate: It removes the information that is no longer required by model.')
st.markdown('c. The output gate: The output Gate at LSTM selects the information to be shown as output This deep learning model has done some work here.')


crypto =st.text_input('Enter crypto Symnbol')
start = '2014-01-01'
end = end = dt.datetime.today()

data = yf.download(crypto, start ,end)

st.subheader('Crypto Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig11=plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'y', label= 'mova_100_days')
plt.plot(data.Close, 'g', label= 'Original data')
plt.xlabel('Time')
plt.ylabel('Price')
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

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price(red) vs Predicted Price(green)')
fig4 = plt.figure(figsize=(10,8))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

real_data=[data_test_scale[len(data_test_scale)-100:len(data_test_scale+1)]]
real_data=np.array(real_data)
real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction=model.predict(real_data)
prediction=scaler.inverse_transform(prediction)
st.write('Next-Day Forecasting')

with st.container():
    col_111, col_222, col_333 = st.columns(3)
    col_111.metric(f'Closing Price Prediction of the next trading day is',
                   f' $ {str(round(float(prediction), 2))}')