import numpy as np
import pandas as pd
from si.data.dataset import Dataset
from statistics import f_classification

class VarianceThreshold:
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.variance = []

    def fit(self, dataset):
        for i in dataset.X:
            self.variance.append(dataset[i].var())
        return self

    def tranform(self):
        X = []
        for x in self.variance:
            if x > self.threshold:
                X.append(x)
        return X
    
    def fit_transform(self, dataset):
        self.fit(dataset)
        self.tranform()
