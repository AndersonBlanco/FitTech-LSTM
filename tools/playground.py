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
import time 

model = keras.saving.load_model('lstm.h5')

num_videos = 1
cap = cv2.VideoCapture(0)
#out = cv2.VideoWriter('output.avi', -1, 20.0, (frame.shape[0], frame.shape[1]))

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
        return 'bad upper cut - rotation lack', pred_y
    elif idx == 8:
        return 'good straight', pred_y
    




f = 0
a = []
text = 'null'

prevTime = 0
newTime = 0 

def runPlayground():

    while True:
        ret, frame = cap.read()

        newTime = time.time()
        fps = 1/(newTime-prevTime) 
        prevTime = newTime 

        #print(fps)

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
            _string, prediction = label(a)
            text = _string 
            #print(_string)
            #print("prediction: ", _string)
            a = []
            f = 0
            newFrame = cv2.putText(newFrame, prediction, ( 10,10 ), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,204), 1)
        

    
        cv2.imshow("Frame", cv2.flip(newFrame))
        #out.write(newFrame)

        if cv2.waitKey(1) == ord('q'):
            break
    
 
    cap.release()
    cv2.destroyAllWindows()
