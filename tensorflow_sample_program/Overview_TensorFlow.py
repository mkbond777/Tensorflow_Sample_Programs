import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

a = tf.add(3, 5)
print(a)

# Session
a = tf.add(3, 5)
sess = tf.Session()
print(sess.run(a))
sess.close()

# with tf.Session()
with tf.Session() as sess:
    print(sess.run(a))

# Complex operations
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    op3 = sess.run(op3)

# Subgraphs
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_up = tf.pow(add_op, mul_op)
with tf.Session as sess:
    z, not_useless = sess.run([pow_up, useless])

# tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)

# It is possible to break graph into several chunks and run them parallelly across mutiple CPUs, GPUs, TPUs

# To put part of a graph on a specific CPU or GPU:

# Create a graph
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0], name='b')
    c = tf.multiply(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

# Graph
g = tf.Graph()
with g.as_default():
    x = tf.add(3, 5)

sess = tf.Session(graph=g)
sess.run(x)


# Create user created graph and default graph separately
# We can use 2 graphs this way but it is never advisable. We should only use 1 graph.
g1 = tf.get_default_graph()
g2 = tf.Graph()
# Adding ops to the default graph
with g1.as_default():
    a = tf.constant(3)
# Adding ops to the user created graph
with g2.as_default():
    b = tf.constant(5)
