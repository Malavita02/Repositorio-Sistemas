import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X, y, features, label):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def get_shape(self):
        return print("shape: ", self.y.shape)
    
    def has_label(self):
        if self.y is None:
            return False
        else: return True
    
    def get_mean(self):
        return np.mean(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {"mean":self.get_mean(),
            "median": self.get.median()}
        )

if __name__ == "__main__":
    Dataset.X = np.array([[1,2,3],[1,2,3]])
    Dataset.y = np.array([1,2,3])
    Dataset.features = None
    Dataset.label = None
    print(Dataset.get_shape)
    print(Dataset.has_label)