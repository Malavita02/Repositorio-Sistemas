import numpy as np
from typing import Callable
from src.si.data.dataset import Dataset
from src.si.statistics.euclidean_distance import euclidean_distance

class KMeans:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.
    Parameters
    ----------
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations.
    distance: Callable
        Distance function.
    Attributes
    ----------
    centroids: np.array
        Centroids of the clusters.
    labels: np.array
        Labels of the clusters.
    """
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        K-means clustering algorithm.
        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.
        """
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        # atributes
        self.centroids = None
        self.labels = None
    
    def get_closest_centroid(self, x):
        distance = self.distance(x, self.centroids)
        closest_index = np.argmin(distance)
        return closest_index
    
    def fit(self, dataset: Dataset) -> 'KMeans':
        seeds = np.random.permutation(dataset.get_shape()[0])[:self.k]
        self.centroids = dataset.X[seeds]

        check = False
        i = 0
        labels = np.zeros(dataset.get_shape()[0])
        while not check and i < self.max_iter:
            new_labels = np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)
            centroids = [np.mean(dataset.X[new_labels == j], axis = 0) for j in range(self.k)]
            self.centroids =np.array(centroids)
            check = np.any(new_labels != labels)
            labels = new_labels
            i += 1
        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        return self.distance(sample, self.centroids)

    def transform(self, dataset):
        centroid_distances = np.apply_along_axis(self._get_distances, axis = 1, arr = dataset.X)
        return centroid_distances

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset):
        return np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)

if __name__ == "__main__":
    from src.si.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)
