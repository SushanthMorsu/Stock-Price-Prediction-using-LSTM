import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# Define start and end dates for data
start = '2015-01-01'
end = date.today().strftime('%Y-%m')

# Streamlit app title
st.title('Stock Price Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data using yfinance
df = yf.download(user_input, start=start, end=end)

# Display basic statistics of the data
st.subheader('Data from 2015 - Today')
st.write(df.describe())

# Plot Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['Close Price'])
st.pyplot(fig)

# Plot Closing Price vs Time Chart with 100-day Moving Average
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(df['Close'], 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Plot Closing Price vs Time Chart with 100-day and 200-day Moving Averages
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.plot(df['Close'], 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Data preprocessing for model training
df1 = df['Close'].values
df1 = df1.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(df1)

# Train-test split
training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:, :]

# Function to create datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]   # i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load pre-trained LSTM model
try:
    model = load_model('keras_model.h5')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Predict train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling for predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE for train and test data
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))

# Display RMSE for train and test
st.subheader('Model Performance Metrics')
st.write(f'Train RMSE: {train_rmse}')
st.write(f'Test RMSE: {test_rmse}')

# Shift predictions for plotting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

# Plot train and test predictions
st.subheader('Train and Test Predictions')
fig = plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df1), 'b', label='Original Price')
plt.plot(trainPredictPlot, 'r', label='Train')
plt.plot(testPredictPlot, 'g', label='Test')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Future predictions (next 30 days)
x_input = test_data[len(test_data) - 100:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = 100
for i in range(30):
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

# Plot next 30 days predictions
day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

st.subheader('Next 30 Days Prediction')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(df1[len(df1) - 100:]), 'b', label='Previous 100 days Price')
plt.plot(day_pred, scaler.inverse_transform(lst_output), 'orange', label='Predicted Price For Next 30 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Final Graph - Visualizing Future Predictions
df3 = df1.tolist()
df3.extend(lst_output)

st.subheader('Visualization of Future Predicted Graph')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df3[1400:])
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['Predicted Future Price'])
st.pyplot(fig3)
