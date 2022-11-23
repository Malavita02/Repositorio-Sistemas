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
        self.final_model.fit(Dataset(dataset.X,predictions.T))
        return self

    def predict(self, dataset: Dataset) -> float:
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        
        return self.final_model.predict(Dataset(dataset.X,predictions.T))

    def score(self, dataset):

        return accuracy(dataset.y, self.predict(dataset))

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

    # initialize final model
    final_model = KNNClassifier(k=3)

    # initialize stacking classifier
    stacking = StackingClassifier([knn, lg], final_model)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    print(stacking.predict(dataset_test))

