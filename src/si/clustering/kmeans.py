import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from statistics.euclidean_distance import euclidean_distance 

class KMeans:
    def __init__(self, k, max_iter, distance = euclidean_distance):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroids = None
        self.labels = None
    
    def get_closest_centroid(self, x):
        distance = self.distance(x, self.centroids)
        closest_index = np.argmin(distance)
        return closest_index
    
    def fit(self, dataset, k):
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

    def transform(self, dataset):
        centroid_distances = np.apply_along_axis(self.distance, axis = 1, arr = dataset.X)
        return centroid_distances

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset):
        return np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)

if __name__ == "__main__":
    a = KMeans(2, 10)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    #a.init_centroids(dataset=dataset)
    print(a.get_closest_centroid([[1, 2, 3, 4], [2, 3, 7, 9]]))