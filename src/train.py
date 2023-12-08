import cv2
import os
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model as load_keras_model

from ai import euclidean_distance, eye_aspect_ratio, save_model, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier,  get_dlib_utils, process_images, load_model, process_images_for_cnn 
from helpers import load_image, cnn_image_size, all_class_labels, load_dataset, cnn_model_path, svc_model_path, knn_model_path, nbc_model_path, cnn_is_greyscale
from more_itertools import flatten

# Train the k nearest neighbors classifier
# 
# Create a kNN classifier instance if one is not provided
# 'n_neighbors' is a hyperparameter that you can tune
def train_knn(feature_vectors, class_labels, classifier = None):
    if classifier is None:
        classifier = load_model(knn_model_path, KNeighborsClassifier(n_neighbors=5))
    
    print("Training the knn model...")
    classifier.fit(feature_vectors, class_labels)
    save_model(classifier, knn_model_path)
    return classifier

# Train the naive bayes classifier
def train_nbc(feature_vectors, class_labels, classifier = None):
    if classifier is None:
        classifier = load_model(nbc_model_path, GaussianNB())

    # Train the model incrementally
    print("Training the nbc model...")
    classifier.partial_fit(feature_vectors, class_labels, classes=all_class_labels)
    save_model(classifier, nbc_model_path)
    return classifier

# Train the support vector classifier
def train_svc(feature_vectors, class_labels, classifier = None):
    if classifier is None:
        new_pipeline = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.01, kernel='rbf'))
        classifier = load_model(svc_model_path, new_pipeline)
    
    print("Training the svc model...")
    classifier.fit(feature_vectors, class_labels)
    save_model(classifier, svc_model_path)
    return classifier

def train_cnn(feature_vectors, class_labels, size, classifier = None):

    if classifier is None:
        try:
            classifier = load_keras_model(cnn_model_path)
        except:
            # Example CNN architecture
            classifier = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 1 if cnn_is_greyscale else 3)),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(len(all_class_labels), activation='softmax')
            ])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training the cnn model...")
    # Convert string labels to integers
    label_encoder = LabelEncoder()
    encoded_class_labels = list(map(lambda x: all_class_labels.index(x), class_labels))
    # integer_encoded = label_encoder.fit_transform(class_labels)

    # save_model(label_encoder, "models/cnn_label_encoder.pkl")

    # convert integers to one-hot encoded labels
    onehot_encoded = to_categorical(encoded_class_labels)

    # Train the model
    #   @param validation_data=(X_val, y_val)
    classifier.fit(feature_vectors, onehot_encoded, epochs=10)

    # Save the model
    classifier.save(cnn_model_path)

model = os.environ['model'] if 'model' in os.environ else ""
data = os.environ['data'] if 'data' in os.environ else ""

training_dir_path = "assets/train" + (data if data != '1' else "" )

print("Training dir path: " + training_dir_path)
classifier = None
dataset = load_dataset(training_dir_path, flatten=True)

print("About to train model: " + model)
if model != "cnn":
    detector, predictor = get_dlib_utils()
    vectors, labels = process_images(dataset, detector, predictor)

    mapped_models = [("knn", train_knn), ("svc", train_svc),  ("nbc", train_nbc)]

    for (model_name, training_fn) in mapped_models:
        if model == model_name or model == "":
            training_fn(vectors, labels)

elif model == "cnn":
    vectors, labels = process_images_for_cnn(dataset, cnn_image_size, cnn_is_greyscale)
    print(vectors)
    train_cnn(vectors, labels, cnn_image_size)
