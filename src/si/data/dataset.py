from cProfile import label
import numpy as np
import pandas as pd
import io


class Dataset:
    def __init__(self, X, y, features, label):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def print_dataset(self):
        df = pd.DataFrame(self.X, columns=self.features)
        return print(df.to_string())

    def get_shape(self):    
        return f"shape: {self.X.shape}"
    
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
            "median": self.get_median(Dataset)}
        )
        
    def remove_na(self):
        df = pd.DataFrame(self.X, columns=self.features)
        df.dropna()
        return df
    
    def replace_na(self, val):
        df = pd.DataFrame(self.X, columns=self.features)
        return df.fillna(val)    

def new_read_csv(filename, sep, features, label):
    return pd.read_csv(filename, delimiter=sep)#, features, label)

def read_data_file(filename, sep, label):
    return np.genfromtxt(filename, delimiter=sep)#,)

if __name__ == "__main__":
    Dataset.X = np.array([[1,np.NAN,3],[1,2,3]])
    Dataset.y = np.array([1,2,3])
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

