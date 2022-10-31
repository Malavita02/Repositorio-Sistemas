import numpy as np
import sys
sys.path.insert(0, 'src/si')
from metrics import accuracy
from data.dataset import Dataset

class VotingClassifier:
    def __init__(self, models) -> list:
        self.models = models
    
    def fit(self, dataset):  
        for model in self.models:
            model.fit(dataset)
        return self
     
    
    def predict(self, dataset: Dataset) -> float:
        #helper functions
        def _get_majority_vote(pred: np.ndarray) -> int:
            labels, counts = np.unique(pred, return_counts = True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).tranpose()
        #podemos nao usar o tranpose e mudar o axis para 0
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)
    
    def score(self, dataset):

        return accuracy(dataset.y, self.predict(dataset))

#falta testar