import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)

xtrain = pd.read_csv('../xtrain3.csv')
xtest = pd.read_csv('../xtest.csv')
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

xtrain = xtrain.values
ytrain = ytrain.values
xtrain = tf.constant(xtrain)
ytrain = tf.constant(ytrain)


m = xtrain.shape[0]
n = xtrain.shape[1]
labels = 1

X = tf.placeholder(tf.float32, [None, n])
Y = tf.placeholder(tf.float32, [None, labels])

theta1 = tf.Variable(tf.random_normal([7, 7]))
theta2 = tf.Variable(tf.random_normal([7]))

b1 = tf.Variable(tf.zeros(7))
b2 = tf.Variable(tf.zeros(1))

Z2 = tf.nn.relu(tf.matmul(X, theta1) + b1)
hypothesis = tf.matmul(Z2, theta2) + b2
learning_rate = 0.001

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

avg_cost = 0
for i in range(m):
    c = sess.run([cost, optimizer], feed_dict={X: xtrain, Y: ytrain})
    avg_cost += c / m
print('m:', '%04d' % (m + 1), 'cost = ', '{:.9f}'.format(avg_cost))