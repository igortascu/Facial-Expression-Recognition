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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model as load_keras_model
from PIL import Image

from ai import euclidean_distance, eye_aspect_ratio, save_model, image_to_landmarks, format_landmarks, get_feature_vector, get_bayes_classifier,  get_dlib_utils, process_images, load_model 
from helpers import load_image, highlight_landmarks, all_class_labels, load_dataset, cnn_model_path, svc_model_path, knn_model_path, nbc_model_path
from more_itertools import flatten

# Train the k nearest neighbors classifier
# 
# Create a kNN classifier instance if one is not provided
# 'n_neighbors' is a hyperparameter that you can tune
def train_knn(feature_vectors, class_labels, classifier):
    if classifier is None:
        classifier = load_model(knn_model_path, KNeighborsClassifier(n_neighbors=5))
    
    print("Training the knn model...")
    classifier.fit(feature_vectors, class_labels)
    save_model(classifier, knn_model_path)
    return classifier

# Train the naive bayes classifier
def train_nbc(feature_vectors, class_labels, classifier):
    if classifier is None:
        classifier = load_model(nbc_model_path, GaussianNB())

    # Train the model incrementally
    print("Training the nbc model...")
    classifier.partial_fit(feature_vectors, class_labels, classes=all_class_labels)
    save_model(classifier, nbc_model_path)
    return classifier

# Train the support vector classifier
def train_svc(feature_vectors, class_labels, classifier):
    if classifier is None:
        new_pipeline = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.01, kernel='rbf'))
        classifier = load_model(svc_model_path, new_pipeline)
    
    print("Training the svc model...")
    classifier.fit(feature_vectors, class_labels)
    save_model(classifier, svc_model_path)
    return classifier

def load_and_preprocess_image(image_path, target_size):
    # Load image
    img = Image.open(image_path)
    
    # Resize image
    img = img.resize(target_size)

    # Convert image to array and normalize
    img_array = np.array(img) / 255.0

    return img_array

def train_cnn(feature_vectors, class_labels, size, model):

    if model is None:
        try:
            model = load_keras_model(cnn_model_path)
        except:
            # Example CNN architecture
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(len(all_class_labels), activation='softmax')
            ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
            #   validation_data=(X_val, y_val)

    print("Training the cnn model...")
    model.fit(feature_vectors, class_labels, epochs=10)

    # Save the model
    model.save(cnn_model_path)

detector, predictor = get_dlib_utils()
training_dir_path = "assets/train"
classifier = None
model = os.environ['model'] if 'model' in os.environ else ""
dataset = load_dataset(training_dir_path, flatten=True, limit=500)

print("About to train model: " + model)
if model != "cnn":
    vectors, labels = process_images(dataset, detector, predictor)
    if model == "knn" or model == "":
        classifier = train_knn(vectors, labels, classifier)
    elif model == "svc" or model == "":
        classifier = train_svc(vectors, labels, classifier)
    elif model == "nbc" or model == "":
        classifier = train_nbc(vectors, labels, classifier)
elif model == "cnn":
    target_size = (64, 64)  # Example size, choose according to your needs

    # Load and preprocess each image
    image_vectors = np.array([load_and_preprocess_image(path, target_size) for path in map(lambda x: x[1], dataset)])
    labels = list(map(lambda x: x[0], dataset))
    train_cnn(image_vectors, labels, target_size, model)
