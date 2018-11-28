import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

xtrain = pd.read_csv('../xtrain3.csv')
xtest = pd.read_csv('../xtest3.csv')
ytest = pd.read_csv('../ytest.csv')
ytrain = pd.read_csv('../ytrain.csv')

# xtrain.Sex = xtrain.Sex.map({'male': 0, 'female': 1})
# xtrain = xtrain.fillna({'Age': 29.0, 'Embarked': 'S'})
# xtrain.Embarked = xtrain.Embarked.map({'S': 0, 'Q': 1, 'C': 2})
# xtrain = xtrain.drop(['Cabin'], axis=1)
# xtrain = xtrain.drop(['PassengerId'], axis=1)
# xtrain = xtrain.drop(['Ticket'], axis=1)
# xtrain = xtrain.drop(['Survived'], axis=1)
# xtrain = xtrain.drop(['Name'], axis=1)
#
# xtest.Sex = xtest.Sex.map({'male': 0, 'female': 1})
# xtest.Embarked = xtest.Embarked.map({'S': 0, 'Q': 1, 'C': 2})
# xtest = xtest.drop(['Cabin'], axis=1)
# xtest = xtest.drop(['PassengerId'], axis=1)
# xtest = xtest.drop(['Ticket'], axis=1)
# xtest = xtest.fillna({'Age': 29.0, 'Fare': 35.6})
# xtest = xtest.drop(['Name'], axis=1)
#
# ytest = ytest.drop('PassengerId', axis=1)

# xtrain = xtrain.values
# ytrain = ytrain.values
# xtrain = tf.constant(xtrain)
# ytrain = tf.constant(ytrain)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
# model.add(tf.keras.layers.Dense(7, input_shape=(7,), activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
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
