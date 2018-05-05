import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)

# Linear equation of form y = 3x + 2
y_data = x_data * 3 + 2

# adding some Guassian Noise
y_data = np.vectorize(lambda y_lamb: y_lamb + np.random.normal(loc=0.0, scale=0.1))(y_data)

# print(list(zip(x_data, y_data))[0:5])

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b

# Square mean error
loss = tf.reduce_mean(tf.square(y - y_data))

# GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Intializing tensorflow global variable.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data = []
for step in range(100):
    # evals = sess.run(train, feed_dict=[a, b])
    evals = sess.run([train, a, b])
    if step % 5 == 0:
        print(step, evals[1:])
        train_data.append(evals[1:])

# Plotting values of a, b which is stored after every 5th iteration.
converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a * x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr, cg, cb))

plt.plot(x_data, y_data, 'ro')

green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()
