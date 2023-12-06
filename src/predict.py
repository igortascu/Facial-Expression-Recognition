import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score
from ai import euclidean_distance, eye_aspect_ratio, image_to_landmarks,process_images, load_model, format_landmarks, get_feature_vector, get_bayes_classifier,  get_dlib_utils
from helpers import load_image, highlight_landmarks, all_class_labels, nbc_model_path, knn_model_path,  load_dataset

detector, predictor = get_dlib_utils()

def predict_with_knn(feature_vectors, classifier):
    print("Predicting with  the knn model...")

    if classifier is None:
          classifier = load_model(knn_model_path)
    return classifier, classifier.predict(feature_vectors)

def predict_with_nbc(feature_vectors, classifier):
    print("Predicting with  the nbc model...")
    if classifier is None:
        classifier = load_model(nbc_model_path)
    return classifier, classifier.predict_proba(feature_vectors)

classifier = None
dataset = load_dataset("assets/predict")
model = os.environ['model'] if 'model' in os.environ else ""

def predict_with_svc(feature_vectors, classifier):
    print("Predicting with  the svc model...")
    if classifier is None:
        classifier = load_model('models/svc.pkl')
    return classifier, classifier.predict(feature_vectors)

accuracy = 0
total = 0

for images in dataset:
    vectors, labels = process_images(images, detector, predictor)
    
    predictions = []
    if model == "knn":
        classifier, predictions = predict_with_knn(vectors, classifier)
    elif model == "svc":
        classifier, predictions = predict_with_svc(vectors, classifier)
    else:
        classifier, predictions = predict_with_nbc(vectors, classifier)
        get_predicted_label = lambda prb: all_class_labels[max(enumerate(prb), key=lambda x: x[1])[0]]
        predictions = list(map(get_predicted_label, predictions))

    for label, predicted in zip(labels, predictions):
        print(label, predicted)
        total += 1
        if predicted == label:
            accuracy += 1
    print("Accuracy: ", accuracy / total)
print("Total Accuracy: ", accuracy / total)

    # print(classification_report(labels, predictions))
    # print("Accuracy:", accuracy_score(labels, predictions))
