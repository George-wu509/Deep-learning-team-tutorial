# ------ This is Deep learning team tutorial example1: MNIST and softmax regression -----

# Download MNIST data from server using input_data.py
# You can use the following link to get input_data.py code:
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/mnist/input_data.py


# Step 1 - MNIST dataset
# Use input_data.py to download MNIST dataset(you should have input_data.py in folder)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Step 2 - Start tensorflow session
# Define sess using interactiveSession() class
import tensorflow as tf
sess = tf.InteractiveSession()

# Step 3 - Define weight and bias initiation functions
#
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# step 4 - Define convolution and pooling functions
#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

# step 5 - First convolutional layer
#
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)












