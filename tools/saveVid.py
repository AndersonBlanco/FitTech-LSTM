import numpy as np
import cv2 as cv
import time
import os
import datetime
import winsound
#input from my camera
time.sleep(10)
winsound.Beep(1000,500)
print('now')
infive = datetime.datetime.now() + datetime.timedelta(0,5)
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object

count = 0

#we need tto use videowritter_fourcc, its our methods of saving vids
fourcc = cv.VideoWriter_fourcc(*'XVID')

#variable meant to be what we output, we use videowriter 
out = cv.VideoWriter('jabsWhereTorsoDoesntRotate1.avi', fourcc, 20.0, (640,  480))

#this section is what saves each frame of the video
#while the camera is opened/ on
while cap.isOpened():
    if count == 100:
        break

    #we do cap.read() which analyzes the state of out current frame

    #if ret is empty exit
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #the current frame is set to flip itself, 0? degrees?
    
    # write the flipped frame

    #we add to out our current frame
    out.write(frame)

    #shows our frame
    cv.imshow('frame', frame)
    count+=1
    #if q is pressed exit
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
