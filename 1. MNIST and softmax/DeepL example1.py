
# ------ This is Deep learning team tutorial example1: MNIST and softmax regression -----

# Download MNIST data from server using input_data.py
# You can use the following link to get input_data.py code:
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/mnist/input_data.py
 
# Use input_data.py to download MNIST dataset(you should have input_data.py in folder)    
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ------ example1: MNIST and softmax regression -----
import tensorflow as tf

# data x, weight W and bias b init values
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Model output y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Using cross entropy to evaluate and train the model
# Definre cross_entropy and use gradient descent to optimize(min) the parameters
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

# Define one session and initializes variables
sess = tf.Session()
sess.run(init)

# Train model 1000 times
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the model   
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Reference: https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html



