import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
tf.enable_eager_execution()

# x = tf.range(-5, 5, 0.1)
# y = tf.div(1.0, tf.add(1, tf.exp(-x)))
# plt.plot(x.numpy(),y.numpy())
# plt.show()
f = open('./a1a.train')
train_raw = f.read()
g = open('./a1a.test')
test_raw = g.read()
def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = np.zeros((num_examples, num_features))
    Y = np.zeros((num_examples, 1))
    for i, line in enumerate(train_lines):
        tokens = line.split()
        label =(int(tokens[0]) + 1)/ 2 # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    X = tf.convert_to_tensor(X, tf.float32)
    Y = tf.convert_to_tensor(Y, tf.float32)
    return X, Y

Xtrain,Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

train_data = tf.data.Dataset((Xtrain, Ytrain), batch_size=64, shuffle=True)
test_data = tf.data.Dataset((Xtrain, Ytrain), batch_size=64, shuffle=True)


