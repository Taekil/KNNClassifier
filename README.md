# K-Nearest Neighbors (KNN) Classifier from Scratch

## Overview
This project implements the K-Nearest Neighbors (KNN) classification algorithm from scratch using Python and NumPy. The goal is to classify data points based on their nearest neighbors and evaluate the accuracy of the classification.

## Features
- Implementation of the KNN algorithm without using external machine learning libraries.
- Support for loading and processing dataset files (CSV format).
- Customizable number of neighbors (K) for classification.
- Computation of Euclidean distance for similarity measurement.
- Evaluation of accuracy by comparing predicted labels with actual test labels.
- Visualization of dataset using Matplotlib.

## How It Works
1. **Data Loading**: The dataset is loaded from CSV files containing labeled feature data.
2. **Training**: The training dataset is stored for future reference (no explicit training process since KNN is a lazy learner).
3. **Prediction**: For each test sample:
   - Compute the Euclidean distance between the test sample and all training samples.
   - Identify the K closest points (neighbors) in the training dataset.
   - Determine the most frequent label among the K nearest neighbors.
   - Assign this label to the test sample as the predicted class.
4. **Evaluation**: The predicted labels are compared with the actual labels from the test set to compute accuracy.

## Implementation Details
### Euclidean Distance Function
Computes the Euclidean distance between two points:
```python
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### KNN Classifier
```python
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```

### Accuracy Calculation
```python
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)
```

## Usage
1. Load the training and test datasets from CSV files.
2. Initialize the KNN classifier with a chosen value of K.
3. Train the classifier using `fit(train_data, train_labels)`.
4. Predict the labels of test samples using `predict(test_data)`.
5. Compute the accuracy of the predictions.

### Example Usage
```python
# Load training data
train_data, train_labels = read_csv('trainYX.csv')

# Load test data
test_data, test_labels = read_csv('testYX.csv')

# Initialize KNN classifier with k=5
clf = KNN(k=5)
clf.fit(train_data, train_labels)

# Predict on test data
predictions = clf.predict(test_data)

# Calculate accuracy
accuracy = calculate_accuracy(test_labels, predictions)
print(f'Accuracy: {accuracy:.4f}')
```

## Visualization
A sample dataset (Iris dataset) is visualized using Matplotlib to illustrate how the KNN classifier distinguishes between different classes.

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scatter plot of the dataset
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()
```

## Explanation of Accuracy Calculation
1. The KNN model predicts the label for each test data point.
2. The predicted labels are compared with the actual test labels.
3. The accuracy is computed as:
   
   \[
   \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Test Samples}}
   \]

4. This provides an evaluation of how well the KNN classifier performs on the given dataset.

## Possible Improvements
- Implementing weighted KNN, where closer neighbors have higher influence.
- Optimizing distance calculations for large datasets.
- Exploring different distance metrics such as Manhattan distance.
- Using KD-Trees or Ball Trees for faster neighbor searches.

## Conclusion
This project demonstrates how the K-Nearest Neighbors algorithm works, providing a hands-on approach to understanding distance-based classification. It can be extended for various applications, including image classification and recommendation systems.

