import os
import sys

venv_path = '/home/sona-inc5619/deep_env/lib/python3.12/site-packages'
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)



import dlib
import cv2 
import numpy as np 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        landmarks = predictor(imgGray,face)
        for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0,255,0), cv2.FILLED)
    
    cv2.imshow('Facial Landmark Detection',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break