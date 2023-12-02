import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
import joblib
from ai import euclidean_distance, eye_aspect_ratio, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier, store_bayes_classifier
from helpers import load_image, highlight_landmarks, all_class_labels
from more_itertools import chunked

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

classifier = get_bayes_classifier()
training_data2 = os.listdir("assets/train2")

store = []

for emotion_label in all_class_labels:
    image_paths = os.listdir("assets/train2/" + emotion_label)
    image_paths_with_labels = map(lambda filename: (emotion_label, f"assets/train2/{emotion_label}/{filename}"), image_paths)
    chunks = iter(list(chunked(image_paths_with_labels, 10)))
    store.append(chunks)

while len(store) > 0:
    for i in range(0, len(store)):
        chunk = next(store[i])
        if chunk:
            print("Training chunk...", chunk)
            train_bayes_classifier(classifier, chunk)

# store_bayes_classifier(classifier)