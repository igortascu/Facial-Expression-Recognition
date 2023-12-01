import cv2

all_class_labels = ["Anger", "Fear", "Happy", "Sad", "Surprised", "Neutral"]

def highlight_landmarks(image, landmarks):
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

def load_image(path):
    image = cv2.imread(path)
    return image