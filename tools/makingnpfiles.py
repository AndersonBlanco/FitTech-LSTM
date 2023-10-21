import numpy as np
import os
import cv2 
import mediapipe as mp
frames = 0
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calc_angle(a,b,c):
     a = np.array(a)
     b = np.array(b)
     c = np.array(c)

     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
     angle = np.abs(radians*180.0/np.pi)

     if angle>180:
         angle = 360-angle
     return angle


cap = cv2.VideoCapture(cv2.samples.findFile('jabsWhereTorsoDoesntRotate1.avi'))
with mp_pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            print('oops')
            break
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try: 
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
                                    )

            Leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Leftelbow= [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            Leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            Lefthip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Leftknee= [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            Leftankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            Rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Rightelbow= [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            Righthip= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Rightknee= [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Rightankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            
                #print("LeftArm", calc_angle(Leftshoulder, Leftelbow, Leftwrist))
                #print("RightArm", calc_angle(Rightshoulder, Rightelbow, Rightwrist))
            angleRightShoulder = calc_angle(Rightelbow, Rightshoulder, Righthip)
            cv2.putText(image, str(round(angleRightShoulder, 2)),
                        tuple(np.multiply(Rightshoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            angleRightElbow = calc_angle(Rightshoulder, Rightelbow, Rightwrist)
            cv2.putText(image, str(round(angleRightElbow, 2)),
                        tuple(np.multiply(Rightelbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            
            angleLeftShoulder = calc_angle(Leftelbow, Leftshoulder, Lefthip)
            cv2.putText(image, str(round(angleLeftShoulder, 2)),
                        tuple(np.multiply(Leftshoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            angleLeftElbow = calc_angle(Leftshoulder, Leftelbow, Leftwrist)
            cv2.putText(image, str(round(angleLeftElbow, 2)),
                        tuple(np.multiply(Leftelbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            
            angleLeftHip = calc_angle(Righthip, Lefthip, Leftknee)
            cv2.putText(image, str(round(angleLeftHip, 2)),
                        tuple(np.multiply(Lefthip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            
            angleRightHip = calc_angle(Lefthip, Righthip, Rightknee)
            cv2.putText(image, str(round(angleRightHip, 2)),
                        tuple(np.multiply(Righthip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            
            angleRightLeg = calc_angle(Righthip, Rightknee, Rightankle)
            cv2.putText(image, str(round(angleRightLeg, 2)),
                        tuple(np.multiply(Rightknee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            
            angleLeftLeg = calc_angle(Lefthip, Leftknee, Leftankle)
            cv2.putText(image, str(round(angleLeftLeg, 2)),
                        tuple(np.multiply(Leftknee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
            angles = [angleRightShoulder,angleRightElbow,angleRightHip,angleRightLeg,angleLeftShoulder,angleLeftElbow,angleLeftHip,angleLeftLeg]
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
                                    )
            frames +=1
            np.save(r'C:\Users\Juan\Desktop\Codes\pythonNumberRec\npy\norotationjab\angles{}'.format(frames), angles)
            cv2.imshow("raw webcam", image)
        except:
            print('no')
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

