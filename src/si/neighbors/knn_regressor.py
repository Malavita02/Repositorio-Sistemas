import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
#falta chamar euclidean_distance com callable
from metrics.rmse import rmse

class KNNRegressor:
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

            # Get the mean value
            knn_means = np.mean(knn_labels)

            return knn_means
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
    def score(self, dataset):
        prediction = self.predict(dataset)
        return rmse(dataset.y, prediction)

#falta testar