import numpy as np
import cv2 as cv
import time
import os
import datetime
import winsound
#input from my camera
time.sleep(10)
#insound.Beep(1000,500)
print('now')
infive = datetime.datetime.now() + datetime.timedelta(0,5)
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object

count = 0

#we need tto use videowritter_fourcc, its our methods of saving vids
fourcc = cv.VideoWriter_fourcc(*'XVID')

num_videos = 10

for x in range(num_videos):
    out = cv.VideoWriter(f'../newData/jab/good/video_{x + 32}.avi', fourcc, 20.0, (640,  480))
    winsound.Beep(1000,500)
    for length in range(40):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        out.write(frame)

 
        cv.imshow('frame', frame)
 
        if cv.waitKey(1) == ord('q'):
            break
    
 
cap.release()
out.release()
cv.destroyAllWindows()
