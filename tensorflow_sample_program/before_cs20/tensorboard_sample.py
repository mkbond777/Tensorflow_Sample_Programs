import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

x = tf.add(a, b)
b = tf.Variable(tf.zeros([3]))

with tf.Session() as sess:
    # below line is for tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(b))

writer.close()
