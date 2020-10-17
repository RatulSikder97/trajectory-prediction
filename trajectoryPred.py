# # Trajectory path prediction
# ## Basic trajectory path prediction using RNN,LSTM
# #### Author: Ratul Sikder
# #### Gmail: ratulsikder104@gmail.com
# ##### Project target: 
# Dataset contain X and Y coordinates  of a projectile of .Here>Golf ball.Here target is to find future path of the projected object.
# ##### Approach:
# @@ Trajectory is the path of projectile motion of object. It can be categetoried as Time Series Analysis.
# @@ So, in this peoject RNN-LSTM model is build with one Input layer, one LSTM hidden layer and one output layer. 
# @@ As activation function, here I use tanh function.
#Import essential modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#dataset import
dataset = pd.read_csv("trag45.csv")
print("Dataset Shape ",dataset.shape)
print("First 5 dataset\n",dataset.head())

#plot unprocessed data
plt.figure(figsize=(20,5))

plt.scatter(dataset['X'][0],dataset['Y'][0],color='orange', label='Initial Position',s=200.0)
plt.scatter(dataset['X'][dataset.shape[0]-1],dataset['Y'][dataset.shape[0]-1],color='g',label='Final Position',s=200.0)

plt.plot(dataset['X'],dataset['Y'],lw=(3.0))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("X-Y graph")
plt.legend()


#select only y coordinates. Because x is linear
selectedData = dataset[['Y']].values
selectedData = selectedData.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(selectedData)


#normalized data plot
plt.figure(figsize=(20,5))

plt.scatter(dataset['X'][0],scaledData[0],color='orange', label='Initial Position',s=200.0)
plt.scatter(dataset['X'][dataset.shape[0]-1],scaledData[-1],color='g',label='Final Position',s=200.0)

plt.plot(dataset['X'],scaledData,lw=(3.0))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("X-Y graph of Scaled data")
plt.legend()

# split into train and test sets
train_size = int(len(scaledData) * 0.67)
test_size = len(scaledData) - train_size

train, test = scaledData[0:train_size,:], scaledData[train_size:len(scaledData),:]

print("Training data size %d\n"%len(train),"Testing data size %d\n"% len(test))


#stateful dataset preparation
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#set lookback to 20 prev output
look_back = 20

#traing data preparation
trainX, trainY = create_dataset(train, look_back)
#test data preparation
testX, testY = create_dataset(test, look_back)

print("Traing data shape dim:", trainX.shape)
print("Test data shape dim:", testX.shape)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0],  look_back,1))
testX = np.reshape(testX, (testX.shape[0],look_back, 1))

# Model
# -- Layer Quantity : 3
# -- Loss Function : Mean Squared Error
# -- Activation Fucntion: tanh(for all layer)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(trainX.shape[1:]),activation='tanh'))
model.add(Dense(1,activation='tanh'))

#compile
model.compile(loss='mean_squared_error', optimizer='adam')
#fit
model.fit(trainX, trainY, epochs=200)
model.summary()

#Results
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Error Score: %.2f RMSE' % (trainScore))
    
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Error Score: %.2f RMSE' % (testScore))


#result plot
plt.figure(figsize=(20,5))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(scaledData)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(scaledData)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaledData)-1, :] = testPredict

# plot baseline and predictions

plt.plot(dataset['X'],scaler.inverse_transform(scaledData),label='Autual Path',lw=5.0,color="darkgray")
plt.plot(dataset['X'],trainPredictPlot,label='Traing Predicted path',lw=3.0,color="red")
plt.plot(dataset['X'],testPredictPlot,label='Test Predicted path',lw=3.0,color="lightgreen")
plt.legend()
plt.scatter(dataset['X'][0],dataset['Y'][0],color='orange', label='Initial Position',s=500.0)
plt.scatter(dataset['X'][dataset.shape[0]-1],scaledData[-1],color='g',label='Final Position',s=500.0)
plt.show()

# !!@Here we see that, model predict the projectile path perfectly with little error.

# !!Here will be show the prediction result with another tragectory dataset :)

#import new data
newData = pd.read_csv("test.csv",usecols=[1])
newData = newData.values
newData = newData.astype('float32')


#normalize dataset
newData = scaler.fit_transform(newData)

#x-y plot
plt.figure(figsize=(15,5))

plt.plot(newData,lw=(3.0),label='Tragectory path')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("X-Y graph")
plt.legend()

#dataset split
train1, test1 = newData[0:train_size,:], newData[train_size:len(newData),:]
print("Training data size %d\n"%len(train1),"Testing data size %d\n"% len(test1))


#traing data preparation
trainX1, trainY1 = create_dataset(train1, look_back)
#test data preparation
testX1, testY1 = create_dataset(test1, look_back)

print("Traing data shape dim:", trainX1.shape)
print("Test data shape dim:", testX1.shape)

# reshape input to be [samples, time steps, features]
trainX1 = np.reshape(trainX1, (trainX1.shape[0],  look_back,1))
testX1 = np.reshape(testX1, (testX1.shape[0],look_back, 1))

trainPredict1 = model.predict(trainX1)
testPredict1 = model.predict(testX1)

# invert predictions
trainPredict1 = scaler.inverse_transform(trainPredict1)
trainY1 = scaler.inverse_transform([trainY1])

testPredict1 = scaler.inverse_transform(testPredict1)
testY1 = scaler.inverse_transform([testY1])

# calculate root mean squared error
trainScore1 = math.sqrt(mean_squared_error(trainY1[0], trainPredict1[:,0]))
print('Train Score: %.2f RMSE' % (trainScore1))

testScore1 = math.sqrt(mean_squared_error(testY1[0], testPredict1[:,0]))
print('Test Score: %.2f RMSE' % (testScore1))


plt.figure(figsize=(15,5))

# shift train predictions for plotting
trainPredictPlot1 = np.empty_like(newData)
trainPredictPlot1[:, :] = np.nan
trainPredictPlot1[look_back:len(trainPredict1)+look_back, :] = trainPredict1
# shift test predictions for plotting
testPredictPlot1 = np.empty_like(newData)
testPredictPlot1[:, :] = np.nan
testPredictPlot1[len(trainPredict1)+(look_back*2)+1:len(newData)-1, :] = testPredict1
# plot baseline and predictions


plt.plot(scaler.inverse_transform(newData),label='Autual Path',lw=5.0,color="darkgray")
plt.plot(trainPredictPlot1,label='Traing Predicted path',lw=3.0,color="red")
plt.plot(testPredictPlot1,label='Test Predicted path',lw=3.0,color="lightgreen")
plt.legend()
plt.show()