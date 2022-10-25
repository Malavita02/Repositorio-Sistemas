from pyexpat import features
import numpy as np

class VarianceThreshold:
    def __init__(self, threshold = 0) -> None:
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
        features = np.array(dataset.features)[mask]
        return Dataset(new_x, dataset.y,list(features),label=None)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src/si')
    from data.dataset import Dataset
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"], label="y")
a = VarianceThreshold()
a = a.fit_transform(dataset)
print(a.features)