import mediapipe as mp
import cv2 
import PIL 
from PIL import Image, ImageOps
import numpy as np

import time 


cap = cv2.VideoCapture(0)
x = 0 
while True:
    ret, frame = cap.read()

    cv2.imshow('Feed', frame)
    cv2.imwrite(f'../newData/imgs/img_{x}', frame)
    x += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()