Real-Time Face Detection using MediaPipe & OpenCV 

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



  
