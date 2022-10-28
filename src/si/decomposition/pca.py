import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, dataset):
        self.mean = np.mean(dataset.X, axis=0)
        self.centred_data = dataset.X - self.mean
        
        U, S, Vt = np.linalg.svd(dataset.X, full_matrices=False)
        self.components = Vt[:self.n_components]
        n = len(dataset.X)
        EV = S**2/(n-1)
        self.explained_variance = EV[:self.n_components]

        return self
    
    def transform(self, dataset):
        U, S, Vt = np.linalg.svd(dataset.X, full_matrices=False)
        V = np.transpose(Vt)
        X_reduced = np.dot(dataset.X, V)

        return X_reduced

    def fit_transform(self,dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = PCA(n_components = 2)
    print(a.fit_transform(dataset=dataset))