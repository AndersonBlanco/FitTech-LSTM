import numpy as np
import cv2 as cv
import time
import os
import datetime
import winsound
from vision import drawSkeleton

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

num_videos = 50

for x in range(num_videos):
    #out = cv.VideoWriter(f'../newData/jab/good/angles/f_{x}.avi', fourcc, 20.0, (640,  480))
    os.makedirs(f'../newData/upper_cut/bad/angles/upper_rotation_lack/vid_{x+50}')
    #cv.waitKey(0) # wait 2 seconds for user to reset
    winsound.Beep(1000,500)
    for y in range(40):
        ret, frame = cap.read()
        angles, newFrame = drawSkeleton(frame)
        np.save(f'../newData/upper_cut/bad/angles/upper_rotation_lack/vid_{x+50}/f_{y}.npy', angles)

        print(f'Frame#{y}: ', angles)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

       # out.write(frame)

 
        cv.imshow('frame', newFrame)
 
        if cv.waitKey(1) == ord('q'):
            break

    
 
cap.release()
#out.release()
cv.destroyAllWindows()
