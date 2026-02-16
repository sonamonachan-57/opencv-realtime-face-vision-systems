import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def dodge(x, y):
    return cv2.divide(x, 255 - y, scale=256)

while True:
    success, img = cap.read()
    if not success:
        break

     
    # Sketch Processing
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGrayInv = 255 - imgGray
    imgBlur = cv2.GaussianBlur(imgGrayInv, (21, 21), sigmaX=5, sigmaY=0)
    finalImg = dodge(imgGray, imgBlur)

    # Convert sketch to 3-channel to stack with original
    finalImgColor = cv2.cvtColor(finalImg, cv2.COLOR_GRAY2BGR)

    # Resize if needed 
    finalImgColor = cv2.resize(finalImgColor, (img.shape[1], img.shape[0]))

    # Combine Original + Sketch
    
    combined = np.hstack((img, finalImgColor))

    cv2.imshow("Original | Sketch", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
