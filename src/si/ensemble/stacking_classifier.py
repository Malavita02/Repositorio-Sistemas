import numpy as np
import sys
sys.path.insert(0, 'src/si')
from metrics import accuracy
from data.dataset import Dataset

class StackingClassifier:
    def __init__(self, models, final_model) -> list:
        self.models = models
        self.final_model = final_model
    
    def fit(self, dataset):  
        for model in self.models:
            model.fit(dataset)
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        self.final_model.fit(predictions)
        return self

    def predict(self, dataset: Dataset) -> float:
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return self.final_model.predict(predictions)

    def score(self, dataset):

        return accuracy(dataset.y, self.predict(dataset))

#falta testar