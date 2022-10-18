from pyexpat import features
import numpy as np

class VarianceThreshold:
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.variance = None

    def fit(self, dataset):
        #variance = Dataset.get_var()
        variance = np.var(dataset.X)
        self.variance = variance
        return self

    def transform(self, dataset):
        mask = self.variance > self.threshold
        new_x = dataset.X[:,mask]
        features = np.array()#falta acabar
        return Dataset(new_x, dataset.y,features)

if __name__ == "__main__":
    from si.data.dataset import Dataset
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                        0, 1, 4, 3],
                        0, 6, 3, 2))
    #acabar o Dataset e testar o init e transform