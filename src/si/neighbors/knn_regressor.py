import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
#falta chamar euclidean_distance com callable
from metrics import accuracy

class KNNRegressor:
    def __init__(self, k, distance) -> float:
        self.k = k
        self.distance = distance
        self.dataset = None
    
    def fit(self, dataset):
        self.dataset = dataset
        return self
    
    def predict(self, dataset):
        pass