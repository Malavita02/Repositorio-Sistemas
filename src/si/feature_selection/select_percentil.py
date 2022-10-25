import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from statistics.f_classification import f_classification 

class SelectPercentil:
    def __init__(self, score_func = f_classification, percentile = 0.25):
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

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.tranform(dataset)

if __name__ == "__main__":
    a = SelectPercentil(0.75)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = a.fit_transform(dataset)
    print(dataset.features)
    print(a.features)
