# Classification Reports

## Models Trained using images in train2 (much larger dataset)

### Dataset 0

#### K-Nearest Neighbors (KNN) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.00      | 0.00   | 1           |
| Fear      | 0.33      | 0.33   | 3           |
| Happy     | 0.75      | 1.00   | 3           |
| Neutral   | 0.50      | 1.00   | 1           |
| Sad       | 0.00      | 0.00   | 1           |
| Surprised | 1.00      | 0.50   | 4           |
| **Accuracy**  |          |        | 0.54         |

#### Support Vector Classifier (SVC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 1.00      | 1.00   | 1           |
| Fear      | 0.67      | 0.67   | 3           |
| Happy     | 0.75      | 1.00   | 3           |
| Neutral   | 1.00      | 1.00   | 1           |
| Sad       | 0.00      | 0.00   | 1           |
| Surprised | 0.75      | 0.75   | 4           |
| **Accuracy**  |          |        | 0.77         |

#### Naive Bayes Classifier (NBC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.00      | 0.00   | 1           |
| Fear      | 0.40      | 0.67   | 3           |
| Happy     | 0.00      | 0.00   | 3           |
| Neutral   | 0.00      | 0.00   | 1           |
| Sad       | 0.00      | 0.00   | 1           |
| Surprised | 0.00      | 0.00   | 4           |
| **Accuracy**  |          |        | 0.15         |

### Dataset 1 (much larger)

### Classification Reports

#### K-Nearest Neighbors (KNN) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.29      | 0.12   | 17          |
| Fear      | 0.36      | 0.26   | 19          |
| Happy     | 0.72      | 0.68   | 19          |
| Neutral   | 0.23      | 0.68   | 19          |
| Sad       | 0.40      | 0.11   | 18          |
| Surprised | 0.50      | 0.28   | 18          |
| **Accuracy**  |          |        | 0.36         |

#### Support Vector Classifier (SVC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.33      | 0.06   | 17          |
| Fear      | 0.50      | 0.16   | 19          |
| Happy     | 0.82      | 0.74   | 19          |
| Neutral   | 0.25      | 0.95   | 19          |
| Sad       | 0.00      | 0.00   | 18          |
| Surprised | 0.64      | 0.39   | 18          |
| **Accuracy**  |          |        | 0.39         |

#### Naive Bayes Classifier (NBC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.50      | 0.06   | 17          |
| Fear      | 0.15      | 0.53   | 19          |
| Happy     | 0.00      | 0.00   | 19          |
| Neutral   | 0.00      | 0.00   | 19          |
| Sad       | 0.00      | 0.00   | 18          |
| Surprised | 0.00      | 0.00   | 18          |
| **Accuracy**  |          |        | 0.10         |

#### Convolutional Neural Network (CNN) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.00      | 0.00   | 18          |
| Fear      | 0.17      | 0.89   | 19          |
| Happy     | 1.00      | 0.05   | 19          |
| Neutral   | 0.00      | 0.00   | 19          |
| Sad       | 0.14      | 0.05   | 19          |
| Surprised | 0.00      | 0.00   | 19          |
| **Accuracy**  |          |        | 0.17         |

## Models Trained using images in train1

### Dataset 1 (same as the trained dataset )

### Classification Reports

#### K-Nearest Neighbors (KNN) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.29      | 0.12   | 17          |
| Fear      | 0.36      | 0.26   | 19          |
| Happy     | 0.72      | 0.68   | 19          |
| Neutral   | 0.23      | 0.68   | 19          |
| Sad       | 0.40      | 0.11   | 18          |
| Surprised | 0.50      | 0.28   | 18          |
| **Accuracy**  |          |        | 0.36         |

#### Support Vector Classifier (SVC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 1.00      | 0.94   | 17          |
| Fear      | 1.00      | 0.95   | 19          |
| Happy     | 0.95      | 1.00   | 19          |
| Neutral   | 0.86      | 1.00   | 19          |
| Sad       | 1.00      | 0.89   | 18          |
| Surprised | 0.94      | 0.94   | 18          |
| **Accuracy**  |          |        | 0.95         |

#### Naive Bayes Classifier (NBC) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.64      | 0.53   | 17          |
| Fear      | 0.16      | 0.26   | 19          |
| Happy     | 0.00      | 0.00   | 19          |
| Neutral   | 0.06      | 0.05   | 19          |
| Sad       | 0.07      | 0.06   | 18          |
| Surprised | 0.00      | 0.00   | 18          |
| **Accuracy**  |          |        | 0.15         |

#### Convolutional Neural Network (CNN) Classification Report
| Emotion   | Precision | Recall | Occurrences |
|-----------|-----------|--------|-------------|
| Anger     | 0.00      | 0.00   | 18          |
| Fear      | 0.17      | 0.89   | 19          |
| Happy     | 1.00      | 0.05   | 19          |
| Neutral   | 0.00      | 0.00   | 19          |
| Sad       | 0.14      | 0.05   | 19          |
| Surprised | 0.00      | 0.00   | 19          |
| **Accuracy**  |          |        | 0.17         |
