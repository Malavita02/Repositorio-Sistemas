import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
<<<<<<< HEAD
from typing import Callable, Union
from statistics.euclidean_distance import euclidean_distance
from metrics import accuracy

class KNNClassifier:
    def __init__(self, k, distance: Callable = euclidean_distance ) -> float:
=======
from statistics.euclidean_distance import euclidean_distance
from metrics.accuracy import accuracy

class KNNClassifier:
    def __init__(self, k, distance = euclidean_distance) -> float:
>>>>>>> dc49363 (23/11)
        self.k = k
        self.distance = distance
        self.dataset = None
    
    def fit(self, dataset):
        self.dataset = dataset
        return self

<<<<<<< HEAD
    def _get_closest_label(self, sample):
=======
    def predict(self, dataset):
        def _get_closest_label(sample):
>>>>>>> dc49363 (23/11)
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

    def predict(self, dataset):
        
        return np.apply_along_axis(_get_closest_label, axis=1, arr=dataset.X)
    
    def score(self, dataset):
        prediction = self.predict(dataset)
        return accuracy(dataset.y, prediction)

<<<<<<< HEAD

# não esta a funcionar
if __name__ == '__main__':
    # import dataset
    from data.dataset import Dataset
=======
if __name__ == '__main__':

>>>>>>> dc49363 (23/11)
    from model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
<<<<<<< HEAD
    print(f'The accuracy of the model is: {score}')
=======
    print(f'The accuracy of the model is: {score}')
>>>>>>> dc49363 (23/11)
