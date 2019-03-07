import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow as tf

train_data=pd.read_csv('./training.csv')
test_data=pd.read_csv('./test.csv')
look_data=pd.read_csv('./idlookupTable.csv')

train_data.describe()
train_data.isnull().any().value_counts()#统计具有缺失值的列
train_data.fillna(method = 'ffill',inplace = True)
train_data.isnull().any().value_counts()


imag=[]
for i in range(0,7049):
    img = train_data['Image'][i].split(' ')
    img=['0' if x=='' else x for x in img]
    imag.append(img)

imag_list=np.array(imag,dtype='float')
x_train=imag_list.reshape(-1,96,96)

plt.imshow(x_train[0],cmap='gray')

plt.show()


training=train_data.drop(['Image'],axis=1)

y_data=[]
for j in range(0,7049):
    y=training.iloc[j,:]
    y_data.append(y)
y_train=np.array(y_data,dtype='float')

from tensorflow import keras
from keras.layers import Dropout,Dense,Flatten

from keras.models import Sequential
model = Sequential([Flatten(input_shape=(96,96)),
Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
Dense(30)])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['mae','accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=32)


#下载测试数据
timag=[]
for i in range(0,1783):
    t_img=test_data['Image'][i].split(' ')
    t_img=['0' if x =='' else x for x in t_img]
    timag.append(t_img)
# timag = []
# for i in range(0,1783):
#     timg = test_data['Image'][i].split(' ')
#     timg = ['0' if x == '' else x for x in img] #当条件为 true 时执行代码，当条件为 false 时执行其他代码
#     timag.append(timg)

timag_list=np.array(imag,dtype='float')
x_test=timag_list.reshape(-1,96,96)

pred=model.predict(x_test)

lookid_list=list(look_data['FeatureName'])
rowid=list(look_data['RowId'])
imageid=list(look_data['ImageId']-1)
pre_list = list(pred)

feature=[]
for f in list(look_data['FeatureName']):
    feature.append(lookid_list.index(f))

preded=[]
for x,y in zip(imageid,feature):
    preded.append(pre_list[x][y])

rowid=pd.Series(rowid,name='RowId')
loc = pd.Series(preded,name = 'Location')
submission=pd.concat([rowid,loc],axis=1)
submission.to_csv('face_key_detection_submission.csv',index = False)























