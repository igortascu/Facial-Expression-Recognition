import cv2
import sys
import os
from more_itertools import chunked, take, flatten as flatten_iter, interleave_longest as interleave

all_class_labels = ["Anger",  
                    "Sad", "Surprised", "Fear","Happy", 
                    "Neutral"]

nbc_model_path = "models/nbc.pkl"
knn_model_path = "models/knn.pkl"
svc_model_path = "models/svc.pkl"
cnn_model_path = "models/cnn.h5"
cnn_image_size = (64, 64)
cnn_is_greyscale = True

def highlight_landmarks(image, landmarks):
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

def load_image(path):
    image = cv2.imread(path)
    return image

# Loads the dataset from the given path
# Inputs:
#   path: The path to the dataset
#   chunks: The number of chunks to split the dataset into. Useful for incremental training
#   flatten: Whether to flatten the dataset or not
#   limit: The maximum number of images to load from each class
def load_dataset(path, chunks=0, flatten=False, limit = 0):
    store = []

    if limit == 0:
        limit = sys.maxsize

    for emotion_label in all_class_labels:
        image_paths = os.listdir(path + "/" + emotion_label)
        image_paths_with_labels = list(map(lambda filename: (emotion_label, f"{path}/{emotion_label}/{filename}"), image_paths))
        
        if chunks == 0:
            if flatten:
                store.extend(take(limit, image_paths_with_labels))
            else: 
                store.append(list(take(limit, image_paths_with_labels)))
            continue
        
        chunks_iter = chunked(take(limit, image_paths_with_labels), chunks)

        if flatten:
            store.extend(flatten_iter(chunks_iter))
        else:
            store.append(list(chunks_iter))

    if chunks == 0:
        return store

    return list(interleave(store))