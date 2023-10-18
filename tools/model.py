import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint
import sklearn
import os
import numpy as np

def getX_getY(path, label):
    x = []
    y = []
    for vid_fiolders in os.listdir(path):
        for frames in os.listdir(path + '/' + vid_fiolders):
            
            data = np.load(path + '/' + vid_fiolders + '/' + frames)
            x.append([data])
            y.append(label)

    return np.array(x), np.array(y)

#[rest, jab, upper_cut]
jab_x, jab_y = getX_getY('../angles/training/jab', [0, 1, 0])
rest_x, rest_y= getX_getY('../angles/training/rest', [1, 0,0])
upper_cut_x, upper_cut_y = getX_getY('../angles/training/upper_cut', [0,0,1])

print(jab_x.shape)
print(jab_y.shape)

x_train, y_train = np.concatenate((jab_x, rest_x, upper_cut_x), axis = 0), np.concatenate((jab_y, rest_y, upper_cut_y), axis = 0)

model = keras.Sequential()
model.add(keras.layers.LSTM(64, return_sequences= True, activation='relu', input_shape = (1, 8)))
model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(keras.layers.LSTM(128, return_sequences=False, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', "accuracy"])

def train_and_save():
    #checkpoint = ModelCheckpoint('lstm.h5', monitor='loss', verbose = 1, save_best_only= True, mode = 'min')
    #callbacks_list = [checkpoint]
    model.fit(x_train, y_train, epochs = 25)

    model.save('lstm.h5')

def predict(angles):
    model = keras.saving.load_model('model.h5')

    prediction = model.predict(angles)

    return prediction

print('starting.....')
train_and_save()
