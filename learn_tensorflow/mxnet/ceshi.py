import tensorflow as tf
x=tf.ones(shape=[3,2])
x=tf.reshape(x,[1,6])
sess=tf.Session()
print (sess.run(x))
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))


