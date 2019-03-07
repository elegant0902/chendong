import numpy as np
import pandas as pd
# import tensorflow as tf

# batch = 200
# all_predict = []

# test_data = pd.read_csv('./test.csv').values.astype(np.float32)
# testing_image = test_data.reshape(-1, 28, 28, 1)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# saver = tf.train.import_meta_graph('./model/-101.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./model'))

# input_data = tf.get_default_graph().get_tensor_by_name('input_data:0')
# fc8 = tf.get_default_graph().get_tensor_by_name('fc8/fc8:0')
# answer = tf.argmax(tf.nn.softmax(fc8), 1)

# for i in range(testing_image.shape[0] // batch):
#     test_output = sess.run(answer, {input_data: testing_image[i * batch : (i + 1) * batch]})
#     test_output = test_output.tolist()
#     all_predict.extend(test_output)
#     print(i)
# pd.to_pickle(all_predict,'answer')

all_predict = pd.read_pickle('answer')
df = pd.DataFrame(all_predict)
df=df.rename(columns={0: "Label"})
df.index = range(1, len(all_predict) + 1)

df.to_csv('asd.csv', index_label=['ImageId'])
