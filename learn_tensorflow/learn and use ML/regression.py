import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

bosten_housing=keras.datasets.boston_housing
(train_data,train_labels),(test_data,test_labels)=bosten_housing.load_data()
order = np.argsort(np.random.random(train_labels.shape))
#np.random.random(1000,20) 1000个浮点数，从0-20；
#np.argsort(),argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
train_data = train_data[order]   
train_labels = train_labels[order]

#示例和功能
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
import pandas as pd
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())
print(train_labels[0:10])

#标准化数据
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(train_data[0])  # First training sample, normalized


#建立模型
def build_model():
    model=keras.Sequential([keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)])
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',#均方损失函数
                optimizer=optimizer,
                metrics=['mae'])
    return model

model = build_model()
model.summary()

#训练模型
# Display training progress by printing a single dot for each completed epoch通过为每个完成的时期打印单个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats存储训练统计信息
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

import matplotlib.pyplot as plt
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)

model=build_model()
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
history=model.fit(train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[early_stop,PrintDot()]
)
plot_history(history)
[loss,mae]=model.evaluate(test_data,test_labels,verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions=model.predict(test_data).flatten()#flatten返回一个折叠成一维的数组
plt.scatter(test_labels,test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')#表示x轴和y轴的单位长度相同
plt.xlim(plt.xlim())#获取或设置当前轴的x范围
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 50)#bin 这个参数指定bin(箱子)的个数,也就是总共有几条条状图
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")




