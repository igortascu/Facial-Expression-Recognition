import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
import joblib
from ai import euclidean_distance, eye_aspect_ratio, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier, store_bayes_classifier
from helpers import load_image, highlight_landmarks, all_class_labels

def predict_emotion(classifier, image_path):
    image = load_image(image_path)
    landmarks = image_to_landmarks(image)
    feature_vector = get_feature_vector(landmarks)
    probabilities = classifier.predict_proba([feature_vector])
    
    probabilities[0] = list(map(lambda x: round(x, 4), probabilities[0]))
    results = list(zip(all_class_labels, probabilities[0]))

    return results

classifier = get_bayes_classifier()

validation_data = os.listdir("assets/predict")
validation_data = list(map(lambda x: int(x), validation_data))
validation_data.sort()

for dir_id in [validation_data[18]]:
    image_paths = os.listdir("assets/predict/" + str(dir_id))
    image_labels_and_paths = []

    for image_path in image_paths:
        for label in all_class_labels:
            if label.lower() in image_path.lower():
                image_labels_and_paths.append((label, "assets/predict/" + str(dir_id) + "/" + image_path))
                
    for (label, path) in image_labels_and_paths:
        print(label + " -> ", predict_emotion(classifier, path))
        