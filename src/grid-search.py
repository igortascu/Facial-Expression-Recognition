from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from helpers import load_dataset
from ai import process_images, get_dlib_utils

'''
    Use techniques like Grid Search with cross-validation to experiment with different combinations of these parameters. 
    Grid Search tests a range of values and finds the combination that performs best according to a given metric.
'''
param_grid = {'svc__C': [0.1, 1, 10, 100],
              'svc__gamma': [1, 0.1, 0.01, 0.001],
              'svc__kernel': ['rbf', 'linear']}

dataset = load_dataset("assets/train2", chunks=60, flatten=True, limit = 360)
detector, predictor = get_dlib_utils()
vectors, labels = process_images(dataset, detector, predictor)

print("Training the grid search model...")
grid_search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), param_grid, cv=5)
grid_search.fit(vectors, labels)

print("Best parameters: ", grid_search.best_params_)

# Results:
# dataset = train: SVC(C=10, gamma=1, kernel='linear')
# dataset = train2: SVC(C=10, gamma=1, kernel='linear')
