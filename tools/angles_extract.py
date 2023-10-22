from vision import drawSkeleton
import os
import cv2 
import numpy as np

path = '../newData/jab/bad/rotation_lack'
count = 0
vid_count = 0


for idx, vid in enumerate(os.listdir(path)):
    vid_path = path + '/' + vid

    cap = cv2.VideoCapture(vid_path)

    vid_count += 1
    os.makedirs(f'../newData/jab/bad/angles/rotation_lack/vid_{vid_count}')

    while cap.isOpened(): 
        ret, frame = cap.read()
        #print(ret)

        if ret == False:
            break 

    
        angles, newFrame = drawSkeleton(frame)
        
        cv2.imshow('Feed', newFrame)
        print(angles)
        np.save(f'../newData/jab/bad/angles/rotation_lack/vid_{vid_count}/f_{count}.npy', angles)
        count += 1
  
        if cv2.waitKey(1) == ord('q'):
            break

        

    cap.release()
    cv2.destroyAllWindows()

