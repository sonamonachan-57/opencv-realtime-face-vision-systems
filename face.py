import os
import sys

venv_path = '/home/sona-inc5619/mp_env/lib/python3.12/site-packages'
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)



import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            print(id, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih)
            )

            cv2.rectangle(img, bbox, (255, 0, 255), 2)

            cv2.putText(
                img,
                f'{int(detection.score[0] * 100)}%',
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2
            )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (20, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        2
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)



# this python code converted to ros2 node

import os
import sys

venv_path = '/home/sona-inc5619/mp_env/lib/python3.12/site-packages'
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class FaceDetector(Node):

    def __init__(self):
        super().__init__('face_detector_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def image_callback(self, msg):
        # Convert ROS Image → OpenCV
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )

                cv2.rectangle(img, bbox, (255, 0, 255), 2)

                cv2.putText(
                    img,
                    f'{int(detection.score[0] * 100)}%',
                    (bbox[0], bbox[1] - 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Face Detection", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    rclpy.spin(node)

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

