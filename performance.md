
## Versions
- v1
  - Actual initial version
  - Trained on a dataset of 1000 images
  - 5 emotions: anger, fear, happy, sad, surprised
  - Feature vectors: distance between inner corner of eyebrows, eyebrow (represents the curve of the eyebrow)

- v1.0.3
- v1.0.4 
  - Updated the dataset by relabeling the images in [train2](./assets/train2/) to better fit the emotion.

- v2
  - Updated featurs to return the entire parametric curve of each curved feature.
  - (trained on v2.1.1)
### Model Performance
#### Naive Bayesian Classifier (nbc)
- v1.0.0 - Sad and Surprised model. For some reason, it classifies all images as either sad or surprised.
- v1.0.1 - Even worse, has no context of anger, fear or happy. It only classifies images as sad or surprised.
- v1.0.2 - Accuracy:  15%
- v1.0.3 - Accuracy:  18%. Same model logic as v1.0.2, but trained on the validation dataset. 
- **v2**
  - **v2.1.1** Accuracy: 18%
  
#### K-Nearest Neighbors (knn)
- **v1.0.2** - Accuracy: 23%. 
  - Trained with the exact same feature vectors as the nbc v1.0.2 model.
- **v1.0.3** - Accuracy: 43%. 
  - Trained with the same dataset as the nbc v1.0.3 model. (validation dataset)
  - Trained with the exact same feature vectors as the nbc v1.0.3 model.
- **v2**
  - **v2.1.1** Accuracy: 42%
  
#### Support Vector Machine (svc)
- **v1.0.2** - Accuracy:  29%.
  - kernel = 'rbf'
  - C = 1.0
  - gamma = '0.01'
- **v1.0.3** - Accuracy:  18%.
  - kernel = 'linear'
  - C = 1.0
  - gamma = '0.1'
- **v2**
  - **v2.1.1** Accuracy: 42%
  
#### Convolutional Neural Network (cnn)
- **v1** - insert the images into the model as is without extracting any features. 
  - **v1.1.1** Accuracy:  32%
  - **v1.2.1** Accuracy:  22%
  - **v1.3.1** Accuracy:  12%

- **v2** - grayscale images
  - **v2.1.1** Accuracy:  20%
  - **v2.2.1** Accuracy:  20%
  - **v2.3.1** Accuracy:  16%

## Classification Reports
### Version 2
#### K-Nearest Neighbors (KNN) Classification Report
| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.00      | 0.00   | 0.00     | 1       |
| Fear      | 0.00      | 0.00   | 0.00     | 3       |
| Happy     | 0.33      | 0.33   | 0.33     | 3       |
| Neutral   | 0.00      | 0.00   | 0.00     | 1       |
| Sad       | 0.00      | 0.00   | 0.00     | 1       |
| Surprised | 0.00      | 0.00   | 0.00     | 4       |
| **Accuracy**  |          |        | 0.08     | 13      |
| **Macro Avg** | 0.06      | 0.06   | 0.06     | 13      |
| **Weighted Avg** | 0.08      | 0.08   | 0.08     | 13      |

#### Support Vector Classifier (SVC) Classification Report
| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.00      | 0.00   | 0.00     | 1       |
| Fear      | 0.00      | 0.00   | 0.00     | 3       |
| Happy     | 0.00      | 0.00   | 0.00     | 3       |
| Neutral   | 0.08      | 1.00   | 0.14     | 1       |
| Sad       | 0.00      | 0.00   | 0.00     | 1       |
| Surprised | 0.00      | 0.00   | 0.00     | 4       |
| **Accuracy**  |          |        | 0.08     | 13      |
| **Macro Avg** | 0.01      | 0.17   | 0.02     | 13      |
| **Weighted Avg** | 0.01      | 0.08   | 0.01     | 13      |

#### Naive Bayes Classifier (NBC) Classification Report
| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.00      | 0.00   | 0.00     | 1       |
| Fear      | 0.00      | 0.00   | 0.00     | 3       |
| Happy     | 0.23      | 1.00   | 0.38     | 3       |
| Neutral   | 0.00      | 0.00   | 0.00     | 1       |
| Sad       | 0.00      | 0.00   | 0.00     | 1       |
| Surprised | 0.00      | 0.00   | 0.00     | 4       |
| **Accuracy**  |          |        | 0.23     | 13      |
| **Macro Avg** | 0.04      | 0.17   | 0.06     | 13      |
| **Weighted Avg** | 0.05      | 0.23   | 0.09     | 13      |

### Basic Feature Extraction