import tensorflow as tf
import keras
from keras.datasets import boston_housing
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

# Get data, print initial shapes of data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print("Training data shape: ", x_train.shape) # (404, 13) 404 houses, 13 attributes
print("Test data shape", x_test.shape) # (102, 13) 102 houses, 13 attributes
print("Training labels shape: ", y_train.shape) # (404, ) 404 prices
print("Test labels shape: ", y_test.shape) # (102, ) 102 prices
print("Max price: ", max(y_train))

# Reformat x's to be number of standard deviations for each attribute
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std

num_attributes = 13
num_price_ranges = 10

model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(num_attributes,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])

history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(history.history.keys())
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

# TODO: Fix error here
print(f'Test loss: {loss:.3}')
print(f'Test mae: {mae:.3}')