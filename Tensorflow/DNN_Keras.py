import pandas as pd
import matplotlib.pyplot as plt
import keras

xtrain = pd.read_csv('../xtrain3.csv')
xtest = pd.read_csv('../xtest3.csv')
ytest = pd.read_csv('../ytest.csv')
ytrain = pd.read_csv('../ytrain.csv')

n = xtest.shape[1]

model = keras.models.Sequential()
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

hist = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=500)

model.predict(xtest)
plt.figure(figsize=(12, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()
