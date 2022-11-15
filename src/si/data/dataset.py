import numpy as np
import pandas as pd
from typing import Tuple, Sequence


class Dataset:
    def __init__(self, X, y = None, features = None, label = None):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def print_dataset(self):
        df = pd.DataFrame(self.X, columns=self.features, index=self.y)
        return print(df.to_string())

    def get_shape(self):    
        return self.X.shape
    
    def has_label(self):
        if self.y is None:
            return False
        else: return True
    
    def get_classes(self):
        return f"dtype: {self.y.dtype}"

    def get_mean(self):
        return np.mean(self.X, axis=0)

    def get_variance(self):
        return np.var(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)
    
    def get_max(self):
        return np.max(self.X, axis=0)
    
    def get_min(self):
        return np.min(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {"mean":self.get_mean(Dataset),
            "median": self.get_median(Dataset)})
        
    def remove_na(self):
        #não é para usar o dropna 
        df = pd.DataFrame(self.X, columns=self.features)
        df.dropna()
        return df
    
    def replace_na(self, val):
        #não é para usar o fillna
        df = pd.DataFrame(self.X, columns=self.features)
        return df.fillna(val)    

    def new_read_csv(self, filename, sep, features, label):
        #ainda não testei
        data = pd.read_csv(filename, sep=sep)
        if features and label:
            self.features = features
            self.label = label
            self.X = data[features]
            self.y = data[label]
            
        elif features and not label:
            self.features = features
            self.label = None
            self.X = data[features]
            self.y = None
        
        elif label and not features:
            self.features = None
            self.label = label
            self.X = None
            self.y = data[label]
            
        else:
            self.features = None
            self.label = None
            self.X = None
            self.y = None

        return Dataset.print_dataset()

    def read_data_file(self, filename, sep, label):
        #ainda não testei
        data = np.genfromtxt(filename, delimiter=sep)
        
        if label:
            self.X = data.loc[1:-1]
            self.y = data.loc[-1] 
            self.features = data.iloc[0][:-1]
            self.label = data.iloc[0][-1]
        
        else:
            self.X = data.loc[1:]
            self.y = None
            self.features = data.iloc[0]
            self.label = None

        return Dataset.print_dataset()
    
    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

if __name__ == "__main__":
    Dataset.X = np.array([[1,np.NAN,3],[1,2,3]])
    Dataset.y = np.array([1,2])
    Dataset.features = np.array(["Ola","Adeus","Ate logo"])
    Dataset.label = np.array("y")
    print(Dataset.get_shape(Dataset))
    print(Dataset.has_label(Dataset))
    print(Dataset.get_classes(Dataset))
    print(Dataset.get_mean(Dataset))
    print(Dataset.get_variance(Dataset))
    print(Dataset.get_median(Dataset))
    print(Dataset.get_max(Dataset))
    print(Dataset.get_min(Dataset))
    print(Dataset.summary(Dataset))
    #Dataset.remove_na(Dataset)
    Dataset.replace_na(Dataset, 0)
    Dataset.print_dataset(Dataset)
    print(Dataset.X)

