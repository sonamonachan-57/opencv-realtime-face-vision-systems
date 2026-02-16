1. Real-Time Face Detection using MediaPipe & OpenCV 

A simple and efficient real-time face detection system built using MediaPipe and OpenCV.
This project detects faces from a live webcam feed and displays bounding boxes with confidence scores and FPS

Technologies Used: 

     Python 3

     OpenCV

     MediaPipe

     NumPy


Face Detection Initialization:

     mpFaceDetection = mp.solutions.face_detection
     faceDetection = mpFaceDetection.FaceDetection()


How It Works:

   Webcam captures real-time video.

  Frames are converted from BGR to RGB (required by MediaPipe).

  MediaPipe Face Detection model processes the image.

  For each detected face:

     Bounding box is calculated

     Confidence score is displayed

     FPS is calculated and shown on screen.

Install dependencies:
             
      pip install opencv-python mediapipe numpy



2. Real Time face blur using opencv:

    This project detects human faces in real-time using OpenCV’s Haar Cascade classifier and applies a Gaussian blur effect to anonymize them.

   It captures video from the webcam, detects faces frame-by-frame, and blurs only the detected face regions while keeping the rest of the frame unchanged.

   
Technologies used:

        Python 3

        OpenCV

        Haar Cascade Classifier (haarcascade_frontalface_default.xml)

How it works:


Capture webcam video using cv2.VideoCapture()

Convert frame to grayscale

Detect faces using detectMultiScale()

Extract Region of Interest (ROI)

Apply cv2.GaussianBlur() to the face

Replace blurred ROI back into original frame

Display processed frame

Output:

![Face blur Output](faceblur.png)


  
