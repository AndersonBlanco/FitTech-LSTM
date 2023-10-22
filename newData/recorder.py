import mediapipe as mp
import cv2 
import PIL 
from PIL import Image, ImageOps
import numpy as np

import time 

from vision import drawSkeleton
cap = cv2.VideoCapture(0)

classes = ['upper_cut']
num_vids_per_class = 5
video_length = 100 
wait = 0

for x in range(len(classes)):
    for y in range((num_vids_per_class)):
        for z in range((video_length)):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)


            #cv2.imshow('Feed', frame)

            if z == 0:
                cv2.putText(frame, 'Pause', (10,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"class: {classes[x]}", (200,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                cv2.imshow('FitTech', frame)
                wait = cv2.waitKey(5000)
                print(wait)
            else:
                angles, frame = drawSkeleton(frame)
                cv2.putText(frame, f"class: {classes[x]}", (200,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                #cv2.imwrite(f'../angles/training/{classes[x]}/video_{y}/f_{z}.png', frame)
                np.save(f'../angles/training/{classes[x]}/video_{y}/f_{z}.npy', angles)
                cv2.imshow('FitTech', frame)


            if cv2.waitKey(1) == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()