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

prevTime = 0
newTime = 0 

#we need tto use videowritter_fourcc, its our methods of saving vids
fourcc = cv.VideoWriter_fourcc(*'XVID')

#variable meant to be what we output, we use videowriter 
out = cv.VideoWriter('setFrameRate1.avi', fourcc, 20.0, (640,  480))

#this section is what saves each frame of the video
#while the camera is opened/ on
"""
while cap.isOpened():
    

    #we do cap.read() which analyzes the state of out current frame
    if count==40:
        winsound.Beep(1000,500)
        count=0
        
    #if ret is empty exit
    ret, frame = cap.read()
    print(ret)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #the current frame is set to flip itself, 0? degrees?
    
    # write the flipped frame

    #we add to out our current frame
    out.write(frame)
    count += 1
    #shows our frame
    cv.imshow('frame', frame)
    
    #if q is pressed exit
    if cv.waitKey(1) == ord('q'):
        break"""

frame_rate = 10
prev = 0
while cap.isOpened():
    
    time_elapsed = time.time() - prev
    res, image = cap.read()
    if time_elapsed > 1./frame_rate:
        print("time elapsed greater")
        newTime = time.time()
        fps = 1/(newTime-prevTime) 
        prevTime = newTime
        print(fps)
        prev = time.time()
    #we do cap.read() which analyzes the state of out current frame
        if count==40:
            #winsound.Beep(1000, 500)
            count=0
            
        #if ret is empty exit
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #the current frame is set to flip itself, 0? degrees?
        
        # write the flipped frame

        #we add to out our current frame
        out.write(frame)
        count += 1
        #shows our frame
        cv.imshow('frame', frame)
        
        #if q is pressed exit
        if cv.waitKey(1) == ord('q'):
            break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()