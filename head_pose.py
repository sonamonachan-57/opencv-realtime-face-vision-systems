import cv2
import dlib
import numpy as np

# Load face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# 3D model points (same as your code)
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get required landmark coordinates
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
        ], dtype="double")

        # Camera matrix
        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        # Project nose direction
        nose_end_point2D, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector, translation_vector,
            camera_matrix, dist_coeffs)

        # Draw landmarks
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        # Draw head pose line
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255,0,0), 2)

    cv2.imshow("Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
