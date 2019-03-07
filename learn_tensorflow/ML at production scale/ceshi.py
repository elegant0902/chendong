import tensorflow as tf 
x = tf.constant([[[ 1,  2,  3],
                  [ 4,  5,  6]],
                 [[ 7,  8,  9],
                  [10, 11, 12]]])

y=tf.transpose(x, perm=[2,0,1])
print(y)
