import numpy as np
from si.data.dataset import Dataset

class SelectPercentil:
    def __init__(self, score_func, percentile):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    
    def fit(self, dataset):
        f, p = self.score_func(dataset)
        return self

    def tranform(self, dataset):
        idxs = np.argsort(self.F)[-self.percentile:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y = dataset.y, features = list(features), label = dataset.label)

    def fit_tranform(self, dataset):
        self.fit(dataset)
        self.tranform(dataset)