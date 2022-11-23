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

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        #podemos nao usar o tranpose e mudar o axis para 0
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)
    
    def score(self, dataset):

        return accuracy(dataset.y, self.predict(dataset))

#Nao esta a funcionar por causa do KNNClassifier

if __name__ == '__main__':
    # import dataset
    from data.dataset import Dataset
    from model_selection.split import train_test_split
    from neighbors.knn_classifier import KNNClassifier
    from linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))