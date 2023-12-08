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

from ai import process_images, load_model, get_dlib_utils, process_images_for_cnn
from helpers import all_class_labels, nbc_model_path, knn_model_path, svc_model_path, load_dataset, cnn_model_path, cnn_image_size, cnn_is_greyscale

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

    probabilities = classifier.predict_proba(feature_vectors)
    get_predicted_label = lambda prb: all_class_labels[max(enumerate(prb), key=lambda x: x[1])[0]]
    predictions = list(map(get_predicted_label, probabilities))

    return classifier, predictions

classifier = None
dataset = load_dataset("assets/predict", flatten=True)
model = os.environ['model'] if 'model' in os.environ else ""

def predict_with_svc(feature_vectors, classifier):
    print("Predicting with  the svc model...")
    if classifier is None:
        classifier = load_model(svc_model_path)
    return classifier, classifier.predict(feature_vectors)

def predict_with_cnn(feature_vectors, model = None):

    print("Predicting with the cnn model...")
    if model is None:
        model = load_keras_model(cnn_model_path)

    predictions = model.predict(feature_vectors)
    decoded_indices = np.argmax(predictions, axis=1)
    predicted_labels = map(lambda x: all_class_labels[x], decoded_indices)

    # label_encoder = load_model("models/cnn_label_encoder.pkl")
    # predicted_labels = label_encoder.inverse_transform(predicted_indices)

    return model, predicted_labels


def print_accuracy(name, expected, predicted):
    print("\nPredicting with the " + name + " model...")
    accuracy = 0
    total = 0
    for expected, predicted in zip(labels, predictions):
        print(expected, predicted)
        total += 1
        if predicted == expected:
            accuracy += 1
    print("Accuracy: ", accuracy / total)
    print("Total Accuracy: ", accuracy / total)


if model != "cnn":
    vectors, labels = process_images(dataset, detector, predictor)

    mapped_models = [("knn", predict_with_knn), ("svc", predict_with_svc),  ("nbc", predict_with_nbc)]

    for (model_name, predict_fn) in mapped_models:
        if model == model_name or model == "":
            classifier, predictions = predict_fn(vectors, classifier)
            print_accuracy(model_name, labels, predictions)

elif model == "cnn":
    vectors, labels = process_images_for_cnn(dataset, cnn_image_size, cnn_is_greyscale)
    classifier, predictions = predict_with_cnn(vectors)
    print_accuracy("cnn", labels, predictions)