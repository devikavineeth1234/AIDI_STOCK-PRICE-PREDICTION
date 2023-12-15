# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

dataset_train.head()

dataset_train.tail()

dataset_train.shape

dataset_train.info()

dataset_train.columns.values

dataset_train.dtypes

dataset_train.describe()

sns.set()
plt.figure(figsize=(8,6))
sns.distplot(dataset_train['High'],color="green")
plt.title("High")
plt.show()

sns.set()
plt.figure(figsize=(8,6))
sns.distplot(dataset_train['Low'],color="red")
plt.title("Low")
plt.show()

plt.savefig('data_box')
plt.show()
import pylab as plot
dataset_train.hist(bins=20,figsize=(25,25))
plt.suptitle('Histogram for each numeric input variable')
plt.savefig('data_hist')
plt.show()

plt.figure(figsize=(5,5))
correlation=dataset_train.corr()
sns.heatmap(correlation, annot=True)
plt.show()

# Feature Scaling
Feature Scaling and Data Preparation:

MinMaxScaler is used to normalize the 'Open' stock prices between 0 and 1.
Creation of input sequences (X_train) and output (y_train) for the LSTM model.
Reshaping the input data for compatibility with the LSTM model.
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60 : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN
Building the RNN Model:

Sequential model initialization.
Addition of four LSTM layers with dropout regularization.
Inclusion of a dense layer with ReLU activation.
Compilation of the model using the Adam optimizer and mean squared error loss.
Fitting the RNN to the training data with 50 epochs and a batch size of 32.

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=64, activation='relu'))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_test.head()

dataset_test.tail()

dataset_test.info()

dataset_test.columns.values

dataset_test.dtypes

dataset_test.describe()

sns.set()
plt.figure(figsize=(8,6))
sns.distplot(dataset_test['High'],color="green")
plt.title("High")
plt.show()

sns.set()
plt.figure(figsize=(8,6))
sns.distplot(dataset_train['Low'],color="red")
plt.title("Low")
plt.show()

plt.figure(figsize=(5,5))
correlation=dataset_test.corr()
sns.heatmap(correlation, annot=True)
plt.show()

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
