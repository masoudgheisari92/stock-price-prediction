
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#create a dataframe
dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')
dataset_train = dataset_train[::-1]  #inverse the data

#convert the dataframe to a numpy array
training_set = dataset_train.iloc[:, 1:2].values

#scale the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#create the scaled training dataset
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
#convert the x_train and y_train to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

#reshape the data to 3 dimentional array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#import machine learning libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#build the LSTM model
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
#last LSTM layer which return sequences is False
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 25))
model.add(Dense(units = 1))

#compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


#train the model
model.fit(X_train, y_train, epochs = 100, batch_size = 32)


#create test dataset
dataset_test = pd.read_csv('tatatest.csv')
dataset_test = dataset_test[::-1]
real_stock_price = dataset_test.iloc[:, 1:2].values


X_test = training_set_scaled[1975:2035]
X_test = np.reshape(X_test,(1,60,1))


# get the model predicted price values
X_test = training_set_scaled[1975:2035]
y_predict = []
for i in range(60,len(real_stock_price)+60+1):
    X_test = np.reshape(X_test,(1,60,1))
    pred = model.predict(X_test)
    y_predict.append(pred)
    X_test = np.append(X_test,pred)
    X_test = np.delete(X_test,0)
    print(X_test)
    

y_predict = np.reshape(y_predict,(-1,1))


predicted_stock_price = sc.inverse_transform(y_predict)

final=[]
for i in predicted_stock_price:
    final.append(i[-1])


#plot the data
plt.figure(figsize=(16,8))
plt.plot(real_stock_price, color = 'blue', label = 'TATA Stock Price')
plt.plot(final, color = 'green', label = 'Predicted TATA Stock Price')

plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()
