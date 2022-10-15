from imp import load_module
from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, scale

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed


import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate

data = yf.download(tickers = '^RUI', start = '2012-03-11',end = '2022-07-10')
data['RSI']=ta.rsi(data.Close,length=14)
data['EMAF']=ta.ema(data.Close, length=20)
data['EMAM']=ta.ema(data.Close, length=100)
data['EMAS']=ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close']-data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['NextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace = True)
data.drop(['Volume', 'Adj Close', 'Date'], axis=1, inplace=True)
data.to_csv("RUI.csv")
#######################################################################################################################################Processing

pre_day=30
scala_x=MinMaxScaler(feature_range=(0,1))
scala_y=MinMaxScaler(feature_range=(0,1))
cols_x=['Open','High','Low','Close','RSI','EMAF','EMAM','EMAS']
cols_y=['NextClose']
scaled_dada_x=scala_x.fit_transform(data[cols_x].values.reshape(-1,len(cols_x)))
scaled_dada_y=scala_y.fit_transform(data[cols_y].values.reshape(-1,len(cols_y)))
x_total = []
y_total = []

for i in range(pre_day, len(data)):
    x_total.append(scaled_dada_x[i-pre_day:i])
    y_total.append(scaled_dada_y[i])

splitlimitX = int(len(x_total)*0.8)
splitlimitY = int(len(y_total)*0.8)

x_train = np.array(x_total[:splitlimitX])
x_test = np.array(x_total[splitlimitX:])
y_train = np.array(y_total[:splitlimitY])
y_test = np.array(y_total[splitlimitY:])
x_total = np.array(x_total)

###################################################################### build Models
lstm_input = Input(shape=(pre_day, 8), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=x_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)
model.save("test.h5")
####################################
model_test=tf.keras.models.load_model("test.h5")
pred_prices=model_test.predict(x_test)
# pred_prices=scala_x.inverse_transform(pred_prices)
# y_test=scala_y.inverse_transform(y_test)


# Plotting the Stat
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(y_test, color="red", label=f"Real RUI Prices")
plt.plot(pred_prices, color="blue", label=f"Predicted RUI Prices", ls='--')
plt.title("RUI Prices")
plt.xlabel("Time")
plt.ylabel("Stock Prices")
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend()
plt.show()






















