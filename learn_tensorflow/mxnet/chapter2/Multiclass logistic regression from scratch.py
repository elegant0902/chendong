import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

x =[]
w = tf.Variable(tf.zeros(784, 10))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = []
cross_entropy = tf.reduce.mean(-tf.reduce_sum(y_ * log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
