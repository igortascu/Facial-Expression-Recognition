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
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model as load_keras_model

from ai import extract_features, load_model, get_dlib_utils, load_dataset, decode_labels_for_cnn
from helpers import all_class_labels, get_model_path, cnn_image_size, cnn_is_greyscale

def predict_with_knn(feature_vectors, version = None, classifier = None):
    if classifier is None:
          path = get_model_path("knn", version)
          classifier = load_model(path)
    return classifier.predict(feature_vectors)

def predict_with_nbc(feature_vectors, version = None, classifier = None):
    if classifier is None:
        path = get_model_path("nbc", version)
        classifier = load_model(path)

    probabilities = classifier.predict_proba(feature_vectors)
    get_predicted_label = lambda prb: all_class_labels[max(enumerate(prb), key=lambda x: x[1])[0]]
    predictions = list(map(get_predicted_label, probabilities))
    return predictions

def predict_with_svc(feature_vectors, version = None, classifier = None):
    if classifier is None:
        path = get_model_path("svc", version)
        classifier = load_model(path)
    return classifier.predict(feature_vectors)

def predict_with_cnn_f(feature_vectors, version = None, model = None):

    print("Predicting with the cnn (features) model...")
    if model is None:
        path = get_model_path("cnn-f", version)
        model = load_keras_model(path)

    if isinstance(feature_vectors, list):
        feature_vectors = np.array(feature_vectors)

    encoded_labels = model.predict(feature_vectors)
    return decode_labels_for_cnn(encoded_labels)

def predict_with_cnn_l(feature_vectors, version = None, model = None):
        print("Predicting with the cnn (landmarks) model...")
        if model is None:
            path = get_model_path("cnn-l", version)
            model = load_keras_model(path)
    
        encoded_labels = model.predict(feature_vectors)
        return decode_labels_for_cnn(encoded_labels)

def predict_with_cnn(feature_vectors, version = None, model = None):

    print("Predicting with the cnn model...")
    if model is None:
        path = get_model_path("cnn", version)
        model = load_keras_model(path)

    if isinstance(feature_vectors, list):
        feature_vectors = np.array(feature_vectors)

    encoded_labels = model.predict(feature_vectors)
    return decode_labels_for_cnn(encoded_labels)

def print_accuracy(name, expected_list, predicted_list):
    print("Classification report for " + name + ": ")
    print(classification_report(expected_list, predicted_list, zero_division=0))

classifier = None
model = os.environ['model'] if 'model' in os.environ else ""
data = os.environ['data'] if 'data' in os.environ else "0"
version = os.environ['v'] if 'v' in os.environ else (os.environ['version'] if 'version' in os.environ else None)

dataset_path = "assets/predict" if data == "0" else "assets/train" + data

if model != "cnn":
    labels, images, landmarks_list = load_dataset(dataset_path)
    vectors = extract_features(landmarks_list)

    mapped_models = [("knn", predict_with_knn), ("svc", predict_with_svc),  ("nbc", predict_with_nbc)]

    for (model_name, predict_fn) in mapped_models:
        if model == model_name or model == "":
            predictions = predict_fn(vectors)
            print_accuracy(model_name, labels, predictions)

elif model == "cnn":
    labels, images, landmarks_list = load_dataset(
        dataset_path, 
        target_size=cnn_image_size, 
        grayscale=cnn_is_greyscale,
        get_landmarks=False
    )
    images = np.array(images)

    predictions = predict_with_cnn(images)
    print_accuracy("cnn", labels, predictions)