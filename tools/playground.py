# Content in this file is mutable and there is no need to keep it consistent. 
# Serves no importance, testing, creative idea development, problem solving and troubleshooting purposes only 

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
         for frames in os.listdir(path + '/' + (vid_fiolders)):
            print(path + '/' + vid_fiolders + '/' + str(frames))
    return np.array(x), np.array(y)

#[rest, jab, upper_cut]
jab_x, jab_y = getX_getY('../angles/training/jab', [0, 1, 0])
#rest_x, rest_y= getX_getY('../angles/training/rest', [1, 0,0])
#upper_cut_x, upper_cut_y = getX_getY('../angles/training/upper_cut', [0,0,1])
