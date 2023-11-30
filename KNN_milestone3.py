import csv
import numpy as np
from collections import Counter


# Function to read data from a CSV file
def read_csv(file_path):
    data = []
    labels = []
    with open(file_path, mode='r') as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            labels.append(int(row[0]))
            # Assuming the image pixels start from the second column onwards
            pixels = np.array([int(pixel) for pixel in row[1:]], dtype=np.uint8)
            data.append(pixels)
    return np.array(data), np.array(labels)


# Function to calculate L2 distance between two images
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


# Load training data
train_data, train_labels = read_csv('trainYX.csv')

# Load test data
test_data, test_labels = read_csv('testYX.csv')

# Initialize KNN classifier
clf = KNN()

# Fit the model
clf.fit(train_data, train_labels)

# Test the model with different values of K
k_values = [1, 3, 5, 7, 9]  # Add more values as needed

# Create a table to report errors (or accuracies)
table = []

for k in k_values:
    # Predict using the test set
    predictions = clf.predict(test_data)

    # Calculate accuracy
    accuracy = calculate_accuracy(test_labels, predictions)

    # Report the error (or accuracy) for each value of K
    table.append([k, accuracy])

# Display the results
for row in table:
    print(f'K = {row[0]}: Accuracy = {row[1]:.4f}')

"""
the returned list -> compared with the label of original(TEST?)
then calculated the accuracy?
for example, 
I can get the emotion label of the each picture of the test data from the closed points and K values.
because we found the closed point from trained data and find the labels for emotion. 
then, the calculated label can be compared with the given label of test data -> this will be check 
the accuracy? of the process.  
"""
