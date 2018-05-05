# coding: utf-8

# <h1 align=center><font size = 5> LOGISTIC REGRESSION WITH TENSORFLOW </font></h1>

# ## Table of Contents
# 
# Logistic Regression is one of most important techniques in data science. It is usually used to solve the classic
# classification problem.
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <font size = 3><strong>Contains:</strong></font>
# <br>
# - <p><a href="#ref1">Linear Regression vs Logistic Regression</a></p>
# - <p><a href="#ref2">Utilizing Logistic Regression in TensorFlow</a></p>
# - <p><a href="#ref3">Training</a></p>
# <p></p>
# </div>
# ----------------

# <a id="ref1"></a>
# ## Logistic Regression
# 
# Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, _y_,
# is categorical. It produces a formula that predicts the probability of the class label as a function of the
# independent variables.
# 
# Despite the name logistic _regression_, it is actually a __probabilistic classification__ model. Logistic
# regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into
#  a probability with the following function:
# 
# $$
# ProbabilityOfaClass = \theta(y) = \frac{e^y}{1+e^y} = exp(y) / (1+exp(y)) = p 
# $$
# 
# which produces p-values between 0 (as y approaches minus infinity) and 1 (as y approaches plus infinity). This now
# becomes a special kind of non-linear regression.
# 
# In this equation, _y_ is the regression result (the sum of the variables weighted by the coefficients),
# `exp` is the exponential function and $\theta(y)$ is the logistics regression, also called logistic curve. It is a
# common "S" shape (sigmoid curve), and was first developed for modelling population growth.
# 
# You might also have seen this function before, in another configuration:
# 
# $$
# ProbabilityOfaClass = \theta(y) = \frac{1}{1+e^{-x}}
# $$
# 
# So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a
# probability:
# 
# <img
# src="https://ibm.box.com/shared/static/kgv9alcghmjcv97op4d6onkyxevk23b1.png", width = "400", align = "center">
# 

# -------------------------------

# <a id="ref2"></a>
# # Utilizing Logistic Regression in TensorFlow
# 
# For us to utilize Logistic Regression in TensorFlow, we first need to import whatever libraries we are going to
# use. To do so, you can run the code cell below.

# In[159]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

my_data = np.genfromtxt('../resources/trainSet_featureScaling_withoutHeader.csv', delimiter=",", dtype=float)
my_test_data = np.genfromtxt('../resources/test_withoutheader.csv', delimiter=",", dtype=float)

# Next, we will load the dataset we are going to use. We separate the dataset into _xs_ and _ys_, and then into
# training _xs_ and _ys_ and testing _xs_ and _ys_, (pseudo-)randomly.

# In[161]:


# Diving data into feature set and label

# training data
my_data_X, my_data_Y = my_data[:, :6], my_data[:, [6]].ravel()
# print(my_data_X[0,:])
my_data_Y_changed = pd.get_dummies(my_data_Y).values
# print(my_data_Y_changed)

# test_date
my_test_X, my_test_Y = my_test_data[:, :6], my_test_data[:, [6]].ravel()
my_test_Y_changed = pd.get_dummies(my_test_Y).values

trainX, testX, trainY, testY = my_data_X, my_test_X, my_data_Y_changed, my_test_Y_changed
print(trainX.shape, testX.shape, trainY.shape, testY.shape)

# In[162]:

# Now we define x and y. These placeholders will hold our data (both the features and label matrices), and help pass
# them along to different parts of the algorithm.
# 
# ### Why use Placeholders?  
# 1) This feature of TensorFlow allows us to create an algorithm which accepts data and knows something about the
# shape of the data without knowing the amount of data going in. <br><br>
# 2) When we insert “batches” of data in training, we can easily adjust how many examples we train on in a single
# step without changing the entire algorithm.

# In[163]:


# numFeatures is the number of features in our input data.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

# ### Set model weights and bias
# We define two TensorFlow variables as our parameters. These variables will hold the weights and biases of our
# logistic regression and they will be continually updated during training.

# In[164]:

# Randomly sample from a normal distribution with standard deviation .01

weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1, numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))

# init_op = tf.global_variables_initializer()
# with tf.Session() as session:
#     session.run(init_op)
#     print(session.run(bias))
#     session.close()

# ###  Logistic Regression model
# 
# We now define our operations in order to properly run the Logistic Regression. 
# 
# However, for the sake of clarity, we can have it broken into its three main components: 
# - a weight times features matrix multiplication operation, 
# - a summation of the weighted features and a bias term, 
# - and finally the application of a sigmoid function. 
# 
# As such, you will find these components defined as three separate operations below.
# 

# In[165]:


# Three-component breakdown of the Logistic Regression equation.
# Note that these feed into each other.
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# <a id="ref3"></a>
# # Training
# 
# The learning algorithm is how we search for the best weight vector (${\bf w}$). This search is an optimization
# problem looking for the hypothesis that optimizes an error/cost measure.
# 
# We will use batch gradient descent which minimizes the cost.
# 
# ### Cost function
# Before defining our cost function, we need to define how long we are going to train and how should we define the
# learning rate.

# In[206]:


# Number of Epochs/iteration in our training
numEpochs = 700

# Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate=0.0001,
                                          global_step=1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)

# In[207]:


# Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")

# Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# Now we move on to actually running our operations.

# In[208]:


# Create a tensorflow session
sess = tf.Session()

# Initialize our weights and biases variables.
init_OP = tf.global_variables_initializer()

# Initialize all tensorflow variables
sess.run(init_OP)

# We also want some additional operations to keep track of our model's efficiency over time. We can do this like so:

# In[209]:


# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# Now we can define and run the actual training loop, like this:

# In[217]:


# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence." % diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # print(step)
        # Report occasional stats
        if i % 50 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # get_correct_prediction_Op
            # get_correct_prediction_Op = sess.run(correct_predictions_OP, feed_dict={X: trainX, yGold: trainY})
            # print(get_correct_prediction_Op.shape)
            # calculate activation_OP
            arg_max = tf.argmax(activation_OP, 1)
            get_activation_OP = sess.run(arg_max, feed_dict={X: trainX})
            # print(get_activation_OP)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            # generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g" % (i, train_accuracy, newCost, diff))

# How well do we perform on held-out test data?
print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                      feed_dict={X: testX,
                                                                 yGold: testY})))
arg_max = tf.argmax(activation_OP, 1)
final_result = sess.run(arg_max, feed_dict={X: testX})
# print(final_result)
# saving output to csv file
np.savetxt("../output/abc.csv", final_result, delimiter=",")

# Cost Plot

# In[657]:

plt.plot([np.mean(cost_values[i - 50:i]) for i in range(len(cost_values))])
plt.show()


# Assuming no parameters were changed, you should reach a peak accuracy of 90% at the end of training,
# which is commendable. Try changing the parameters such as the length of training, and maybe some operations to see
# how the model behaves. Does it take much longer? How is the performance?

# ------------------------------------
