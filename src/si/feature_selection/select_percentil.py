import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    """
    Select features according to the percentile highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.
    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float, default=0.25
        Percentile of top features to select.
    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.25):
        """
        Select features according to the percentile highest score.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float, default=0.25
            Percentile of top features to select.
        """
        # parameters
        self.score_func = score_func
        self.percentile = percentile
        # attributes
        self.F = None
        self.p = None
    
    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def tranform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the percentile highest scoring features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-int(len(self.F)*self.percentile):]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y = dataset.y, features = list(features), label = dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the percentile highest scoring features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.tranform(dataset)

if __name__ == "__main__":
    a = SelectPercentile(percentile= 0.75)
    """
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    """
    # testing using iris.csv dataset
    from si.io.csv import read_csv
    dataset = read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\iris.csv", features=True, label= True)
    a = a.fit_transform(dataset)
    print(dataset.features)
    print(a.features)
