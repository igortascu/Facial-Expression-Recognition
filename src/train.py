import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
import joblib
from ai import euclidean_distance, eye_aspect_ratio, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier, store_bayes_classifier
from helpers import load_image, highlight_landmarks, all_class_labels

def train_bayes_classifier(classifier, image_labels_and_paths):
    class_labels = []
    feature_vectors = []

    for (label, path) in image_labels_and_paths:
        image = load_image(path)
        if image is None:
            print('Could not open or find the image: ');
            with open("error.txt", "a") as f:
                f.write('Could not open or find the image: ' + path + "\n")
            continue

        landmarks = image_to_landmarks(image)

        if landmarks is None:
            print("Could not find face in image: " + path)
            with open("error.txt", "a") as f:
                f.write("Could not find face in image: " + path + "\n")
            continue

        feature_vector = get_feature_vector(landmarks)
        class_labels.append(label)
        feature_vectors.append(feature_vector)

    # Train the model incrementally
    print("Training the model...")
    classifier.partial_fit(feature_vectors, class_labels, classes=all_class_labels)
    store_bayes_classifier(classifier)

training_data = os.listdir("assets/train")
training_data = list(map(lambda x: int(x), training_data))
training_data.sort()

classifier = get_bayes_classifier()

for image_id in training_data:
    emotions = os.listdir("assets/train/" + str(image_id))
    image_labels_and_paths = []

    for emotion in emotions:
        for label in all_class_labels:
            if label in emotion:
                image_labels_and_paths.append((label, "assets/train/" + str(image_id) + "/" + emotion))
                
    print(image_labels_and_paths)
    train_bayes_classifier(classifier, image_labels_and_paths)

store_bayes_classifier(classifier)