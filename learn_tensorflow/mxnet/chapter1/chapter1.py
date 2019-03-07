# #问题1   初始化1*256的数组，并返回最大值的索引
import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(1)
x=tf.constant(0.0,shape=[1,256])
print(tf.argmax(x,1))


#问题2  生成一个4*4的均匀矩阵与4*4的对角线为1的矩阵相乘
a=tf.random_uniform(shape=[4,4])
b=tf.diag([1.0,1.0,1.0,1.0])
c=tf.matmul(a,b)
# c=tf.matmul(a,b)
sess=tf.Session()  
X,Y=sess.run((a,c))

print(sess.run((a,c))) 
print(sess.run(b))
print(sess.run(c))   #输出tensor方法一
# sess=tf.InteractiveSession() 
# print(a.eval())  
# print(b.eval())  
# print(c.eval())   #输出tensor方法二

#问题3  创建一个3x3x20张量，使得在每个x，y坐标处，在z坐标中移动会列出斐波那契序列。因此，在z的0位置，3x3矩阵将全部为1。在z位置1处，3x3矩阵将全部为1。在z位置2处，3x3矩阵将全部为2，在z位置3处，3x3矩阵将全部为3s，依此类
import tensorflow as tf
tf.enable_eager_execution()
x=tf.ones(shape=[3,3,2])
for i in range(2,20):
    y=x[:,:,i-1]+x[:,:,i-2]
    y=tf.reshape(y,[3,3,1])
    x=tf.concat([x,y],-1)
print(x)

#问题4   求矢量的和与均值
import tensorflow as tf
x=tf.ones(shape=[3,2])
x=tf.reshape(x,[1,6])
sess=tf.Session()
print (sess.run(x))
print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_mean(x)))
#问题5   求解两适量之间的角度

