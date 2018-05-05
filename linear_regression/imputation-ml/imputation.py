import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

my_data = np.genfromtxt('../resources/15pass-normalization.csv', delimiter=",", dtype=float)

# getting features and labels
my_data_X, my_data_Y = my_data[:, :6], my_data[:, [6]].ravel()

# get dummies
my_data_Y_changed = pd.get_dummies(my_data_Y).values

# training and testing data
trainX, testX, trainY, testY = train_test_split(my_data_X, my_data_Y_changed, test_size=0.33, random_state=42)

# numFeatures is the number of features in our input data.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
numLabels = trainY.shape[1]

X = tf.placeholder(tf.float32, shape=numFeatures)
Y = tf.placeholder(tf.float32, shape=numLabels)

# tf.Variable call creates a  single updatable copy in the memory and efficiently updates
# the copy to relfect any changes in the variable values through out the scope of the tensorflow session
m = tf.Variable(3.0)
c = tf.Variable(2.0)

# Construct a Model
Ypred = tf.add(tf.multiply(X, m), c)

# create session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# get prediction with initial parameter values
pred = session.run(Ypred, feed_dict={X: trainX})

# plot initial prediction against datapoints
plt.plot(trainX, pred)
plt.plot(trainX, trainY, 'ro')
# label the axis
# plt.xlabel("# Chirps per 15 sec")
# plt.ylabel("Temp in Farenhiet")

# Loss function
loss = tf.reduce_mean(tf.square(trainY - pred))

# GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

convergenceTolerance = 0.0001
previous_m = np.inf
previous_c = np.inf

steps = {'m': [], 'c': []}

losses = []

for k in range(100000):
    ########## Your Code goes Here ###########
    _, _m, _c, _l = session.run([train, m, c, loss], feed_dict={X: trainX, Y: trainY})
    steps['m'].append(_m)
    steps['c'].append(_c)
    losses.append(_l)
    if (np.abs(previous_m - _m) <= convergenceTolerance) or (np.abs(previous_c - _c) <= convergenceTolerance):
        print("Finished by Convergence Criterion")
        print(k)
        print(_l)
        break
    previous_m = _m,
    previous_c = _c,

session.close()
