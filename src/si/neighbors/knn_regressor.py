import numpy as np
from src.si.data.dataset import Dataset
from typing import Callable, Union
from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.metrics.rmse import rmse

class KNNRegressor:
    """
    KNN Regressor
    The k-Nearst Neighbors regressor is a machine learning model that predicts values of new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.
    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use
    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance
        # attributes
        self.dataset = None
    
    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """
        It returns the closest label of the given sample
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of
        Returns
        -------
        label: str or int
            The closest label
        """
        # Calculates the distance between the samples and the dataset
        distances = self.distance(sample, self.dataset.X)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]
        # Get the mean value
        knn_means = np.mean(knn_labels)

        return knn_means
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """

        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
    def score(self, dataset: Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        prediction = self.predict(dataset)
        return rmse(dataset.y, prediction)

if __name__ == '__main__':
    from src.si.model_selection.split import train_test_split
    from src.si.io.csv import read_csv
    # import dataset cpu.csv
    dataset = read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\cpu.csv",
                       features=True, label=True)
    # split dataset
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
    # initialize the KNN regressor
    knn = KNNRegressor(k=3)
    # fit the model to the train dataset
    knn.fit(dataset_train)
    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')