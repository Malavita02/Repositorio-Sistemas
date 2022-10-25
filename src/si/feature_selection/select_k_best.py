import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
#from typing import Callable
from statistics.f_classification import f_classification 


class SelectKBbest:
    def __init__(self, score_func = f_classification, k = 10) -> None:
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
    
    def fit(self, dataset):
        f, p = self.score_func(dataset)
        return self

    def transform(self,dataset):
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y = dataset.y, features = list(features), label = dataset.label)

    def fit_tranform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    a = SelectKBbest(k = 3)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.fit(dataset)
    b = a.transform(dataset)
    print(b.features)