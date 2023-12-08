import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
import joblib
from ai import euclidean_distance, eye_aspect_ratio, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier,  get_dlib_utils
from helpers import load_image, highlight_landmarks, all_class_labels

detector, predictor = get_dlib_utils()

def predict_emotion(classifier, image_path):
    image = load_image(image_path)
    if image is None: return None

    landmarks = image_to_landmarks(image, detector, predictor)
    if landmarks is None: return None
    feature_vector = get_feature_vector(landmarks)
    probabilities = classifier.predict_proba([feature_vector])
    
    probabilities[0] = list(map(lambda x: round(x, 4), probabilities[0]))
    results = list(zip(all_class_labels, probabilities[0]))

    return results

classifier = get_bayes_classifier()

validation_data = os.listdir("assets/predict")
validation_data = list(map(lambda x: int(x), validation_data))
validation_data.sort()

accuracy = 0
total = 0

for dir_id in validation_data:
    image_paths = os.listdir("assets/predict/" + str(dir_id))
    labelled_images = []

    for image_path in image_paths:
        for label in all_class_labels:
            if label.lower() in image_path.lower():
                labelled_images.append((label, "assets/predict/" + str(dir_id) + "/" + image_path))
    
    for (label, path) in labelled_images:
        results = predict_emotion(classifier, path)
        if results is None:
            print("Could not predict_emotion for image: " + path)
            continue
        predicted_label = max(results, key=lambda x: x[1])[0]
        total += 1
        
        if predicted_label == label:
            accuracy += 1

        print(label + " -> ", results, " (acc = ", accuracy / total, ")")
        
print("Accuracy: ", accuracy / total)