import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing

log_dir='log/'
batch = 32
keep_prob = 0.5
max_step=600
init_lr=0.001
decay_rate = 0.1
repeat_times = 10
model_dir='model/'

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

train_data = train_data.values.astype(np.float32)
np.random.shuffle(train_data)

train_data1 = train_data[:, 1:]
training_data = train_data1[:41900, :]
validing_data = train_data1[41900:, :]

train_data2 = train_data[:, :1]
training_label = train_data2[:41900, :]
validing_label = train_data2[41900:, :]

OneHotEncoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
OneHotEncoder.fit(training_label)
training_label = OneHotEncoder.transform(training_label).toarray()
validing_label = OneHotEncoder.transform(validing_label).toarray()

training_image = training_data.reshape(-1, 28, 28,1)
validing_image = validing_data.reshape(-1, 28, 28,1)

def inference(images):
    parameters = []
    
    with tf.name_scope('conv1') as scope:
        '''
        train_image:28*28*3
        kernel:2*2*96
        sride:1*1
        padding:name

        input:train_image[28*28*3]
        middle:conv1[27*27*96]
        output:pool1[25*25*96]
        '''
        kernel1 = tf.Variable(tf.truncated_normal([2, 2, 1, 96], stddev=0.01,dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(images, kernel1, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], name='biases'))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print(conv1.op.name,'',conv1.get_shape().as_list())
        parameters += [kernel1, biases]
        
        #添加LRN层和max_pool层
        lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool1')
        print(pool1.op.name,'',pool1.get_shape().as_list())

    with tf.name_scope('conv2') as scope:
        kernel2 = tf.Variable(tf.truncated_normal([3, 3, 96, 256], stddev=0.01, dtype=tf.float32), name='weight')
        conv=tf.nn.conv2d(pool1,kernel2,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]), name='baises')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print(conv2.op.name,'',conv2.get_shape().as_list())
        parameters += [kernel2, biases]

        lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print(pool2.op.name,'',pool2.get_shape().as_list())

    with tf.name_scope('conv3') as scope:
        kernel3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01, dtype=tf.float32), name='weight')
        conv=tf.nn.conv2d(pool2,kernel3,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384]), name='baises')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print(conv3.op.name,'',conv3.get_shape().as_list())
        parameters += [kernel3, biases]

    with tf.name_scope('conv4') as scope:
        kernel4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01, dtype=tf.float32), name='weight')
        conv=tf.nn.conv2d(conv3,kernel4,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384]), name='baises')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print(conv4.op.name,'',conv4.get_shape().as_list())
        parameters += [kernel4, biases]

        
    with tf.name_scope('conv5') as scope:
        kernel5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01, dtype=tf.float32), name='weight')
        conv=tf.nn.conv2d(conv4,kernel5,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]), name='baises')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print(conv5.op.name,'',conv5.get_shape().as_list())
        parameters += [kernel5, biases]

        lrn5 = tf.nn.lrn(conv5, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn5')
        pool5 = tf.nn.max_pool(lrn5, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        print(pool5.op.name, '', pool5.get_shape().as_list())

    with tf.name_scope('fc6') as scope:
        '''
        input:pool5 [6*6*256]
        output:fc6 [4096]
        '''
        kernel6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.1, name='weight'))
        biases = tf.Variable(tf.constant(0.0, shape=[4096], name='biases'))
        flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc = tf.nn.relu(tf.matmul(flat, kernel6) + biases, name='fc6')
        fc6 = tf.nn.dropout(fc, keep_prob)
        parameters += [kernel6, biases]
        print(fc6.op.name, '', fc6.get_shape().as_list())


    with tf.name_scope('fc7') as scope:
        '''
        input:fc6[4096]
        output:fc7 [4096]
        '''
        kernel7 = tf.Variable(tf.truncated_normal([4096,4096], stddev=0.1, name='weight'))
        biases = tf.Variable(tf.constant(0.0, shape=[4096],name='biases'))
        fc = tf.nn.relu(tf.matmul(fc6, kernel7) + biases, name='fc7')
        fc7 = tf.nn.dropout(fc, keep_prob)
        parameters += [kernel7, biases]
        print(fc7.op.name, '', fc7.get_shape().as_list())

    with tf.name_scope('fc8') as scope:
        '''
        input:fc7[4096]
        output:fc8[10]
        '''
        kernel8 = tf.Variable(tf.truncated_normal([4096,10], stddev=0.1, name='weight'))
        biases = tf.Variable(tf.constant(0.0, shape=[10], name='biases'))
        fc8 = tf.nn.xw_plus_b(fc7, kernel8, biases, name='fc8')
        parameters += [kernel8, biases]
        print(fc8.op.name, '', fc8.get_shape().as_list())
    return fc8, parameters

def main():
    with tf.Graph().as_default():
        input_data=tf.placeholder(tf.float32,[None,28,28,1],name='input_data')
        input_label=tf.placeholder(tf.float32,[None,10],name='input_label')
        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        
        learning_rate=tf.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

        fc8, parameters = inference(input_data)

        loss=tf.losses.softmax_cross_entropy( input_label,fc8)

        minimize = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name='minimize')
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(fc8), 1), tf.argmax(input_label, 1)), tf.float32))

        saver = tf.train.Saver(max_to_keep=0,filename='alexnet')
        sess = tf.Session()
        if tf.train.latest_checkpoint(model_dir):
            saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        else:
            sess.run(tf.global_variables_initializer())
        
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merge_all = tf.summary.merge_all()
        FileWriter = tf.summary.FileWriter(log_dir, sess.graph)

        num =  training_image.shape[0]// batch
        for i in range(max_step*repeat_times//batch+1):
            temp_train = training_image[i % num * batch:i % num * batch + batch,:]
            temp_label = training_label[i % num * batch:i % num * batch + batch,:]
            sess.run(minimize, feed_dict={input_data: temp_train, input_label: temp_label})
            if sess.run(global_step) % 100 == 1:
                summary = sess.run(merge_all, feed_dict={input_data:validing_image,input_label:validing_label})
                FileWriter.add_summary(summary, sess.run(global_step))
                saver.save(sess, model_dir, global_step)
            print(sess.run(global_step))

if __name__=="__main__":
    main()



