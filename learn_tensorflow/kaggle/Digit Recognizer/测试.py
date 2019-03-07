import tensorflow as tf


a = tf.Variable(tf.truncated_normal([3, 3, 96, 256], stddev=0.01, dtype=tf.float32), name='weight')
print(a.shape)