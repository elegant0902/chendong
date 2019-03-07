import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./training.csv')
df.dropna(inplace=True)
df.shape

from joblib import Parallel, delayed

def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    x = ex(delayed(format_img)(e) for e in df.Image)
    
x = np.stack(x)[..., None]
print(x.shape)
y = df.iloc[:, :-1].values
print(y.shape)

def show(x, y=None):
    plt.imshow(x[..., 0], 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
        
    plt.axis('off')

sample_idx = np.random.choice(len(x))    
show(x[sample_idx], y[sample_idx])

#图像正则化
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)
x_train_norm=x_train[:,:,:,:]
x_val_norm=x_val[:,:,:,:]
x_train_norm=x_train_norm.reshape([1712,96*96,1])
x_val_norm=x_val_norm.reshape([428,96*96,1])
mu=x_train_norm.mean()
sigma=x_train_norm.std()
x_train_norm=(x_train_norm - mu)/sigma
x_val_norm=(x_val_norm - mu)/sigma
x_train_norm=x_train_norm.reshape([1712,96,96,1])
x_val_norm=x_val_norm.reshape([428,96,96,1])
show(x_train_norm[15], y_train[15])

#建模型
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import regularizers

model=Sequential([
    Conv2D(72,4,input_shape=(96,96,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)),
    AvgPool2D(pool_size=(2,2)),
    Conv2D(48,2,activation='relu',use_bias=False,kernel_initializer='he_normal' ,kernel_regularizer=regularizers.l2(0.01)), #Según clase, no se debe inicializar bias antes de un batchnorm
    BatchNormalization(),
    Flatten(),
    Dropout(0.5), #Actúa como regularizador
    Dense(48,activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)), #Importante utilizar he initialization para relu
#     Dropout(0.2), #Actúa como regularizador
#     Dense(40,activation='relu', kernel_initializer='he_normal'), #Importante utilizar he initialization para relu
    Dense(30, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)) # No hay activación acá por ser un problema de regresión
])

model.compile(optimizer=Adam(0.01),loss='mse',metrics=['mae']) # Settings según indicaciones
model.summary(), model.input, model.output
log=model.fit(x_train_norm, y_train, epochs=150, batch_size=256, validation_data=[x_val_norm,y_val])
print(f'MAE final: {model.evaluate(x_val, y_val)[1]}')
print(f'MAE final: {model.evaluate((x_val-mu)/sigma, y_val)[1]}')
print(f'MAE final: {model.evaluate(x_val_norm, y_val)[1]}')
def show_results(*logs):
    trn_loss, val_loss, trn_acc, val_acc = [], [], [], []
    
    for log in logs:
        trn_loss += log.history['loss']
        val_loss += log.history['val_loss']
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(trn_loss, label='train')
    ax.plot(val_loss, label='validation')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend()
    
show_results(log)