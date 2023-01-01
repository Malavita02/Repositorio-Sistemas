import numpy as np
from src.si.data.dataset import Dataset

class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.
    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """
    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        # parameters
        self.threshold = threshold
        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.
        Returns
        -------
        self : object
        """
        variance = np.var(dataset.X, axis=0)
        self.variance = variance
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset
        Returns
        -------
        dataset: Dataset
        """
        mask = self.variance > self.threshold
        new_x = dataset.X[:,mask]
        features = np.array(dataset.features)[mask]
        return Dataset(new_x, dataset.y,list(features),label=None)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        Parameters
        ----------
        dataset: Dataset
        Returns
        -------
        dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"], label="y")
selector = VarianceThreshold()
selector = selector.fit(dataset)
dataset = selector.transform(dataset)
print(dataset.features)