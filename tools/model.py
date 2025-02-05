import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint
import sklearn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def getX_getY(path, label):
    x = []
    y = []

    for vid_fiolders in os.listdir(path):
        for angles in os.listdir(path + '/' + (vid_fiolders)):
                data = np.load(path + '/' + vid_fiolders + '/' + str(angles))
                x.append(data)
                y.append(label)
    return np.array(x), np.array(y)


def prev_org():
     
    #[rest, jab, upper_cut]
    jab_x, jab_y = getX_getY('../angles/training/jab', [0, 1, 0])
    rest_x, rest_y= getX_getY('../angles/training/rest', [1, 0,0])
    upper_cut_x, upper_cut_y = getX_getY('../angles/training/upper_cut', [0,0,1])

    print(jab_x.shape)
    print(jab_y.shape)

    x_train, y_train = np.concatenate((jab_x, rest_x, upper_cut_x), axis = 0), np.concatenate((jab_y, rest_y, upper_cut_y), axis = 0)

    #x_train.resize(4, 100, 8)
    #y_train.resize(4, 100, 3)

    print(x_train.shape)
    print(y_train.shape)





#!IMPORTANT: output shape definition = [good_jab, bad_jab_knee_lvl, bad_jab_rotation]
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.

# Add a LSTM layer with 128 internal units.
model.add(tf.keras.Input((40, 8)))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.LSTM(32))

# Add a Dense layer with 10 units.
model.add(tf.keras.layers.Dense(8))


model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['categorical_accuracy', "accuracy"])

def train_and_save(x, y, xt, xy):
    #checkpoint = ModelCheckpoint('lstm.h5', monitor='loss', verbose = 1, save_best_only= True, mode = 'min')
    #callbacks_list = [checkpoint]
    model.fit(x, y, batch_size=16, epochs = 50)

    #model.save('lstm.h5')

    '''pred_y= model.predict(xt)
    pred_fixed=[]
    for i in range(0, len(pred_y)):
        if pred_y[i].max() == pred_y[i][0]:
            pred_fixed.append(0)
        elif pred_y[i].max() == pred_y[i][1]:
            pred_fixed.append(1)
        elif pred_y[i].max() == pred_y[i][2]:
            pred_fixed.append(2)
    print(accuracy_score(pred_fixed, xy))'''
    #print(pred)


def predict(angles):
    model = keras.saving.load_model('lstm.h5')

    prediction = model.predict(angles)

    print(prediction)

#print('starting.....')

#0 = good
#1 = bad jab - knee_lvl_
#2 = bad jab - rotation
#3 = good rest
#4 = bad rest
#5 = good upper_cut
#6 = bad upper_cut - knee lvl
#7 = bad upper_cut - rotation lack
#8 = good straight 

#jab - good  
x, y = getX_getY('../newData/jab/good/angles', 0)
x.resize(100, 40, 8)
y.resize(100)

#jab - bad knee_lvl_lack
x2, y2 = getX_getY('../newData/jab/bad/angles/knee_lvl_lack', 1)
x2.resize(100, 40, 8)
y2.resize(100)

#jab - bad rotation_lack
x3, y3 = getX_getY('../newData/jab/bad/angles/rotation_lack', 2)
x3.resize(100, 40, 8)
y3.resize(100)


#rest - good
x4, y4 = getX_getY('../newData/rest/good/angles', 3)
x4.resize(100, 40, 8)
y4.resize(100)

#rest - bad
x5, y5 = getX_getY('../newData/rest/bad/angles', 4)
x5.resize(100, 40, 8)
y5.resize(100)


#upper_cut - good
x6, y6 = getX_getY('../newData/upper_cut/good/angles', 5)
x6.resize(100, 40, 8)
y6.resize(100)


#upper_cut - bad - knee lvl
x7, y7 = getX_getY('../newData/upper_cut/bad/angles/upper_knee_lvl_lack', 6)
x7.resize(100, 40, 8)
y7.resize(100)

#upper_cut - bad - rotation lack
x8, y8 = getX_getY('../newData/upper_cut/bad/angles/upper_rotation_lack', 7)
x8.resize(100, 40, 8)
y8.resize(100)

#good straight
x9, y9 = getX_getY('../newData/straight_right/good/angles', 8)
x9.resize(100, 40, 8)
y9.resize(100)

#x4,x5,x6,x7,x8, x9
#y4,y5,y6,y7,y8, y9
x = np.concatenate((x,x2,x3, x4,x5,x6,x7,x8, x9 ), axis=0)
y = np.concatenate((y,y2,y3, y4,y5,y6,y7,y8. y9), axis=0)
print(y)
#print(x.shape)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2)

#print(x_train.shape)
#print(x_train.shape)

#jab - bad - knee_lvl

#train_and_save(x_train, y_train, x_test, y_test)


#predict(x_test[0])
