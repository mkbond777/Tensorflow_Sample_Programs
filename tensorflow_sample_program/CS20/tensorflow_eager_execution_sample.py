import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfe.enable_eager_execution()

i = tf.constant(0)
while i < 1000:
    i = tf.add(i, 1)
print("I could do this all day! %d" % i)

x = tf.constant([1.0, 2.0, 3.0])

# Tensor are backed by NumPy arrays
assert type(x.numpy()) == np.ndarray

# Tensor are iterable!
for j in x:
    print(j)

# Automatic differentiation is built into eager execution

# Operations are recorded on a tape
# The tape is played back to compute gradients
# This is reverse-mode differentiation (backpropagation).

# APIs for computing gradients work even when eager execution is not enabled

# tfe.gradients_function()
# tfe.value_and_gradients_function()
# tfe.implicit_gradients()
# tfe.implicit_value_and_gradients()

# But, when eager execution is enabled  …
# prefer tfe.Variable under eager execution (compatible with graph construction)
# manage your own variable storage — variable collections are not supported!
# use tf.contrib.summary
# use tfe.Iterator to iterate over datasets under eager execution
# prefer object-oriented layers (e.g., tf.layers.Dense)
# functional layers (e.g., tf.layers.dense) only work if wrapped in tfe.make_template
# prefer tfe.py_func over tf.py_func


# Use eager if you're :

# a researcher and want a flexible framework
# python control flow and data structures enable experimentation
# developing a new model
# immediate error reporting simplifies debugging
# new to TensorFlow
# eager execution lets you explore the TF API in the Python REPL

# 2nd and 3rd Last slide contains useful information about tensorflow.
