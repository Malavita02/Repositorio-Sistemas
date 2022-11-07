import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
#falta chamar euclidean_distance com callable
from metrics import accuracy

class KNNClassifier:
    def __init__(self, k, distance) -> float:
        self.k = k
        self.distance = distance
        self.dataset = None
    
    def fit(self, dataset):
        self.dataset = dataset
        return self

    def predict(self, dataset):
        def _get_closest_label(self, sample):
            # Calculates the distance between the samples and the dataset
            distances = self.distance(sample, self.dataset.X)

            # Sort the distances and get indexes
            knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
            knn_labels = self.dataset.y[knn]

            # Returns the unique classes and the number of occurrences from the matching classes
            labels, counts = np.unique(knn_labels, return_counts=True)

            # Gets the most frequent class
            high_freq_lab = labels[np.argmax(counts)]  # get the indexes of the classes with the highest frequency/count

            return high_freq_lab
        
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
    def score(self, dataset):
        prediction = self.predict(dataset)
        return accuracy(dataset.y, prediction)
