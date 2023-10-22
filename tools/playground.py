# Content in this file is mutable and there is no need to keep it consistent. 
# Serves no importance, testing, creative idea development, problem solving and troubleshooting purposes only 

import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint
import sklearn
import os
import numpy as np
import cv2 
import winsound
from vision import drawSkeleton

model = keras.saving.load_model('lstm.h5')

num_videos = 1
cap = cv2.VideoCapture(0)

def label(angles):
    pred_y = np.array(model.predict(angles))
    idx = pred_y[0].argmax(axis = 0)
    
    if idx == 0:
        return 'good jab', pred_y
    elif idx == 1:
        return 'bad jab - knee lvl lack', pred_y
    elif idx == 2:
        return 'bad jab - rotation lack', pred_y
    elif idx == 3:
        return 'good rest', pred_y
    elif idx == 4:
        return 'bad rest', pred_y
    elif idx == 5:
        return 'good upper cut', pred_y
    elif idx == 6:
        return 'bad upper cut - knee lvl lack', pred_y
    elif idx == 7:
        return 'bad jab - rotation lack', pred_y
    




f = 0
a = []
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    angles, newFrame = drawSkeleton(frame)
    
    if f != 40:
        f += 1
        a.append(angles)
    else:
        winsound.Beep(1000,500)
        a = np.array(a)
        a.resize(1,40,8)
        s, p = label(a)
        #cv2.putText(frame, p, (10,10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        print(s)
        print(p)
        a = []
        f = 0
        
 
    cv2.imshow('frame', newFrame)


    if cv2.waitKey(1) == ord('q'):
        break
    
 
cap.release()
cv2.destroyAllWindows()
