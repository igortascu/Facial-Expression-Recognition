import cv2
import dlib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.naive_bayes import GaussianNB
import joblib
from helpers import all_class_labels

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Euclidean distance
# Shortest distance between two points
def euclidean_distance(p1: Point, p2: Point):
    dx = abs(p1.x - p2.x)
    dy = abs(p1.y - p2.y)
    return np.sqrt(dx**2 + dy**2)


def image_to_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    # Detect faces
    faces = detector(image)

    # Only one face is expected in the image
    if len(faces) == 0:
        return None
    
    # Find the landmarks
    landmarks = predictor(image, faces[0])

    return landmarks

def format_landmarks(landmarks):
    def extract(landmarks, start, end):
        return [Point(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]
    
    return {
        'jaw': extract(landmarks, 0, 17), # 17 is not included
        'left_eyebrow': extract(landmarks, 17, 22),
        'right_eyebrow': extract(landmarks, 22, 27),
        'nost_bridge': extract(landmarks, 27, 31),
        'nost_tip': extract(landmarks, 31, 36),
        'left_eye': extract(landmarks, 36, 42),
        'right_eye': extract(landmarks, 42, 48),
        'outer_lips': extract(landmarks, 48, 60),
        'inner_lips': extract(landmarks, 60, 68),
        'inner_upper_lip': extract(landmarks, 60, 65),
        'inner_lower_lip': extract(landmarks, 64, 68) + extract(landmarks, 60, 60 + 1), # 60 is the left corner of the inner lip

    }

# Compute the eye aspect ratio
# This is the ratio between the width and the height of the eye
# link - https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = euclidean_distance(eye[0], eye[3])

    # Compute the eye aspect ratio
    ratio = (A + B) / (2.0 * C)

    return ratio

# Determine the signature of the given points of a curved line
def curve_vectors(curved):
    # `points` is an array of x and y coordinates of the eyebrow points
    points = np.array([[p.x, p.y] for p in curved])

    # Fit a B-spline to the points
    tck, u = splprep([points[:,0], points[:,1]], s=0, k=4)
    u_new = np.linspace(u.min(), u.max(), 1000)
    # x_new, y_new = splev(u_new, tck, der=0) # for visualization where x_new and y_new are arrays of x and y coordinates of the spline

    # Calculate the first derivatives
    dx, dy = splev(u_new, tck, der=1)

    # Calculate the second derivatives
    d2x, d2y = splev(u_new, tck, der=2)

    # Calculate the curvature
    # formula shows the curvature of a parametric curve
    curvature = (dx * d2y - dy * d2x) / np.power(dx**2 + dy**2, 3/2)
    
    # Calculate the maximum curvature
    max_curvature = np.max(np.abs(curvature))

    # Calculate the mean curvature
    mean_curvature = np.mean(np.abs(curvature))

    # Calculate the area under the curvature curve (integral)
    area_under_curve = np.trapz(np.abs(curvature), u_new)

    ## Could add the distance between the highest and the line connecting the two ends of the eyebrow
    return [area_under_curve, max_curvature, mean_curvature]

def mouth_aspect_ratio(mouth):
    # Compute the ratio between the width and the height of the mouth

    # Compute the euclidean distances between the three sets of vertical mouth landmarks
    A = euclidean_distance(mouth[1], mouth[7])
    B = euclidean_distance(mouth[2], mouth[6])
    C = euclidean_distance(mouth[3], mouth[5])

    # Compute the euclidean distance between the horizontal
    D = euclidean_distance(mouth[0], mouth[4])

    ratio = (A + B + C) / (2.0 * D)

    return ratio
def get_bayes_classifier():
    # Load existing model if it exists
    # Otherwise create a new model
    try:
        classifier = joblib.load('models/naive_bayes_classifier.pkl')
    except FileNotFoundError:
        classifier = GaussianNB()

    return classifier

def store_bayes_classifier(classifier):
    joblib.dump(classifier, 'models/naive_bayes_classifier.pkl')

def get_feature_vector(landmarks):
    landmarks = format_landmarks(landmarks)
    feature_vector = []

    left_eye_aspect_ratio = eye_aspect_ratio(landmarks['left_eye'])
    feature_vector.append(left_eye_aspect_ratio)

    right_eye_aspect_ratio = eye_aspect_ratio(landmarks['right_eye'])
    feature_vector.append(right_eye_aspect_ratio)

    left_eyebrow_curve_vectors = curve_vectors(landmarks['left_eyebrow'])
    feature_vector.extend(left_eyebrow_curve_vectors)

    right_eyebrow_curve_vectors = curve_vectors(landmarks['right_eyebrow'])
    feature_vector.extend(right_eyebrow_curve_vectors)

    distance_between_eyebrows = euclidean_distance(landmarks['left_eyebrow'][4], landmarks['right_eyebrow'][0])
    feature_vector.append(distance_between_eyebrows)

    mouth_corner_distance = euclidean_distance(landmarks['outer_lips'][0], landmarks['outer_lips'][6])
    feature_vector.append(mouth_corner_distance)

    inner_lips_aspect_ratio = mouth_aspect_ratio(landmarks['inner_lips'])
    feature_vector.append(inner_lips_aspect_ratio)

    inner_upper_lip_curve_vectors = curve_vectors(landmarks['inner_upper_lip'])
    feature_vector.extend(inner_upper_lip_curve_vectors)

    inner_lower_lip_curve_vectors = curve_vectors(landmarks['inner_lower_lip'])
    feature_vector.extend(inner_lower_lip_curve_vectors)

    return feature_vector
