import cv2
from PIL import Image
import numpy as np
import sys
import os
import joblib
from more_itertools import take 

all_class_labels = ["Anger",  
                    "Sad", "Surprised", "Fear","Happy", 
                    "Neutral"]

nbc_model_path = "models/nbc.pkl"
knn_model_path = "models/knn.pkl"
svc_model_path = "models/svc.pkl"
cnn_model_path = "models/cnn.h5"
cnn_image_size = (64, 64)
cnn_is_greyscale = True

def get_defualt_model_path(name):
    return f"models/{name}.pkl"

def get_model_path(name, version = None):
    ext = "h5" if "cnn" in name else "pkl"

    if version:
        return f"models/{name}/{version}.{ext}"
    
    if os.path.islink(f"models/{name}.{ext}"):
        return os.readlink(f"models/{name}.{ext}")
    
    return f"models/{name}.{ext}"

def highlight_landmarks(image, landmarks):
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

def load_image(path, target_size = None, grayscale = False, bgr = False, normalize = False):
    # Load image using Pillow
    pil_image = Image.open(path)

    if target_size:
        # Resize image
        pil_image = pil_image.resize(target_size)

    print("grayscale", grayscale)
    if grayscale:
        # Convert to grayscale
        pil_image = pil_image.convert('L')

    # Convert PIL image to RGB (if not already in this format)
    # This step should be after resizing and grayscale conversion
    pil_image = pil_image.convert('RGB')
    img = np.array(pil_image)

    if normalize:
        # Normalize the image
        img = img / 255.0

    # Convert RGB to BGR for dlib (because dlib uses BGR)
    if bgr:
        img = img[:, :, ::-1]

    return img

# Loads the label and full path for each image in the dataset
#
# Inputs:
#   path: The path to the dataset
#   flatten: Whether to flatten the dataset or not
#   limit: The maximum number of images to load from each class
def load_dataset_paths(path, limit = 0):
    store = []

    if limit == 0:
        limit = sys.maxsize

    for emotion_label in all_class_labels:
        image_paths = os.listdir(path + "/" + emotion_label)
        image_paths_with_labels = list(map(lambda filename: (emotion_label, f"{path}/{emotion_label}/{filename}"), image_paths))
        
        store.extend(take(limit, image_paths_with_labels))

    return store
