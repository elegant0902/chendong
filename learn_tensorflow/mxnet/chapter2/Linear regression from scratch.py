import tensorflow as tf 
tf.enable_eager_execution()
num_examples=100
x=tf.random_normal(shape=[num_examples,2])
w=tf.constant([2.0,-3.4],shape=[2,1])
b=tf.constant(4.2,shape=[num_examples,1])
n=0.1*tf.random_normal(shape=[num_examples,1])
y1=tf.matmul(x,w)
y2=tf.add(y1,b)
y3=tf.add(y2,n)
print(y3)

import matplotlib.pyplot as plt 
plt.scatter(x[:,0].numpy(),y3.numpy())
plt.show()

def get_train_data(data_length):
    train_arr=[]
    for i in range(data_length):
        tr_x=tf.random_uniform(0.0,1.0)
        tr_y=tf.matmul(va_x,W)+b+tf.random.uniform(-0.02,0.02)  
        train_arr.append([tr_x,tr_y])
    return train_arr

def get_validate_data(data_length):
    validate_arr=[]
    for i in range(data_length):
       va_x=tf.random.uniform(0.0,1.0)        
       va_y=tf.matmul(va_x,W)+b+tf.random.uniform(-0.02,0.02)  
       validate_arr.append([va_x,va_y])      
    return validate_arr
traindata=get_train_data(200)
trainx=[v[0] for v in traindata]
trainy=[v[1] for v in traindata]
plt.plot(trainx,trainy,'ro',label='train data')

plt.show()
Y=[]
W=tf.Variable(tf.random_normal([1]),name='weight')

b=tf.Variable(tf.random_normal([1]),name='bias')
y3=tf.matmul(x,w)+b
losst=tf.reduce_mean(tf.square(y3-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.009)
train=optimizer.minimize(cost)


