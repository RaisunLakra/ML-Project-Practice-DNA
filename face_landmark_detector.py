import sys

REQUIRED_PACKAGES = ['cv2', 'dlib', 'face_recognition', 'face_recognition_models']

missing = []

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print('Please install the correct virtual environment.')
    print('Missing: {}'.format(', '.join(missing)))
    print('Correct environment: ~/internship\\ project/face-recognition-opencv/face-recog-env/bin/activate')
    sys.exit(1)

import cv2
import dlib
import face_recognition_models

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor_path = face_recognition_models.pose_predictor_model_location()
predictor = dlib.shape_predictor(predictor_path)
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()