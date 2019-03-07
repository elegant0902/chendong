import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import sleep
import os

Train_Dir = './training.csv'
Test_Dir = './test.csv'
lookid_dir = './IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)  
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)

train_data.describe()
train_data.isnull().any().value_counts()#统计具有缺失值的列
train_data.fillna(method = 'ffill',inplace = True)
train_data.isnull().any().value_counts()

imag = []
for i in range(0,7049):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img] #当条件为 true 时执行代码，当条件为 false 时执行其他代码
    imag.append(img)

#重置数据类型
image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96)

#显示图像
plt.imshow(X_train[0],cmap='gray')
plt.legend()
plt.show()

#分离标签
training = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,7049):
    y = training.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')

from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential

model = Sequential([Flatten(input_shape=(96,96)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae','accuracy'])
model.fit(X_train,y_train,epochs =2,batch_size = 128,validation_split = 0.2)

timag = []
for i in range(0,1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in img] #当条件为 true 时执行代码，当条件为 false 时执行其他代码
    timag.append(timg)

timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96)
plt.imshow(X_test[0])
plt.show()

#预测图形
pred = model.predict(X_test)

#递交结果
lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)
rowid = lookid_data['RowId']
rowid=list(rowid)

feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))

# imageID=pd.read_pickle('imageID')
# feature=pd.read_pickle('feature')
# pre_list=pd.read_pickle('pre_list')
# rowid=pd.read_pickle('rowid')
# set(imageID)
preded = []
for x,y in zip(imageID,feature):
    try:
        preded.append(pre_list[x][y])
    except:
        print(x)
        print(y)

rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('face_key_detection_submission.csv',index = False)