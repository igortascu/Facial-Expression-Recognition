import cv2
import dlib
import numpy as np

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# Load the image
image = cv2.imread('assets/face.png')

if image is None:
    print('Could not open or find the image');
    exit(0)

print("image", image)

# Detect faces
faces = detector(image)

print("faces", faces)

for face in faces:
    landmarks = predictor(image, face)
    print("landmarks", landmarks)

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


cv2.imwrite('assets/face.out.png', image)
