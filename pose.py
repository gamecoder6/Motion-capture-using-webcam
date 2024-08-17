import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
count = 0
alldata = []
fps_time = 0

pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
              'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY',
              'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
              'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']


cap = cv2.VideoCapture("I:\motion capture\\jj1_nvKjgVcZ.mp4")

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()


        
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_out = np.copy(image)
        image = np.zeros(image.shape)
        """mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)"""
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        #  if(results.pose_landmarks is not None and results.left_hand_landmarks is not None and results.right_hand_landmarks is not None):
        if results.pose_landmarks:
            data_tubuh = {}
            for i in range(len(pose_tubuh)):
                results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image.shape[0]
                results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image.shape[1]
                data_tubuh.update(
                    {pose_tubuh[i]: results.pose_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)

        """if results.right_hand_landmarks:
            data_tangan_kanan = {}
            for i in range(len(pose_tangan)):
                results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x * image.shape[0]
                results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y * image.shape[1]
                data_tubuh.update(
                    {pose_tangan[i]: results.right_hand_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)"""

        """if results.left_hand_landmarks:
            data_tangan_kiri = {}
            for i in range(len(pose_tangan)):
                results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x * image.shape[0]
                results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y * image.shape[1]
                data_tubuh.update(
                    {pose_tangan_2[i]: results.left_hand_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)"""

        #cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, )
        cv2.imshow('MediaPipe Holistic', image)  
        cv2.imshow('Skelton structure', image_out)
        count = count + 1
        print(count)
        fps_time = time.time()
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            df = pd.DataFrame(alldata)
            df.to_excel('I:\motion capture\coordinator3.xlsx')
            break
cap.release()
cv2.destroyAllWindows()
