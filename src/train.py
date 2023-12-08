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

from ai import euclidean_distance, eye_aspect_ratio, encode_labels_for_cnn, save_model, image_to_landmarks, load_dataset, format_landmarks, get_feature_vector, get_bayes_classifier,  get_dlib_utils, extract_features, load_model
from helpers import load_image, cnn_image_size, get_model_path, all_class_labels, cnn_model_path, svc_model_path, knn_model_path, nbc_model_path, cnn_is_greyscale
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
    return classifier

# Train the naive bayes classifier
def train_nbc(feature_vectors, class_labels, classifier = None):
    if classifier is None:
        classifier = load_model(nbc_model_path, GaussianNB())

    # Train the model incrementally
    print("Training the nbc model...")
    classifier.partial_fit(feature_vectors, class_labels, classes=all_class_labels)
    return classifier

# Train the support vector classifier
def train_svc(feature_vectors, class_labels, classifier = None):
    if classifier is None:
        new_pipeline = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.01, kernel='rbf'))
        classifier = load_model(svc_model_path, new_pipeline)
    
    print("Training the svc model...")
    classifier.fit(feature_vectors, class_labels)
    return classifier

def train_cnn_on_features(feature_vectors, class_labels, classifier = None):

    print("Training the cnn (features) model...")
    num_features = len(feature_vectors[0])

    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(len(all_class_labels), activation='softmax')
    ])

    encoded_labels = encode_labels_for_cnn(class_labels)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    if isinstance(feature_vectors, list):
        feature_vectors = np.array(feature_vectors)

    # Train the model
    model.fit(feature_vectors, encoded_labels, epochs=10, batch_size=32)

    return model

def train_cnn_on_images(feature_vectors, class_labels, size, classifier = None):

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
    encoded_labels = encode_labels_for_cnn(class_labels)

    # Train the model
    #   @param validation_data=(X_val, y_val)
    classifier.fit(feature_vectors, encoded_labels, epochs=10)

    return classifier

model = os.environ['model'] if 'model' in os.environ else ""
data = os.environ['data'] if 'data' in os.environ else "1"
output = os.environ['output'] if 'output' in os.environ else None

training_dir_path = "assets/train" + data

print("Training dir path: " + training_dir_path)

if model != "cnn":
    labels, images, landmarks_list = load_dataset(training_dir_path)
    vectors = extract_features(landmarks_list)

    mapped_models = [("knn", train_knn), ("svc", train_svc), ("nbc", train_nbc)]

    for (model_name, training_fn) in mapped_models:
        if model == model_name or model == "":
            model_logic = training_fn(vectors, labels)
            
            if output:
                print(f"Saving {model_name} model to " + output)

                output_path = get_model_path(model_name, output)
                save_model(model_logic, output_path)

                default_path = f"models/{model_name}.pkl"
                
                try:
                    os.remove(default_path)
                except:
                    pass

                os.symlink(output_path, default_path)
            else:
                if os.path.exists(f"models/{model_name}.pkl"):
                    os.remove(f"models/{model_name}.pkl")
                save_model(model_logic, f"models/{model_name}.pkl")

elif model == "cnn":
    labels, images, landmarks_list = load_dataset(training_dir_path)
    images = np.array(images)

    train_cnn_on_images(images, labels, cnn_image_size)
