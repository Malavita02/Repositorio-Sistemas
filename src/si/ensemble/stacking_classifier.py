import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset

class StackingClassifier:
    """
    Stacked generalization consists in stacking the output of individual estimator
    and use a classifier to compute the final prediction.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    final_model : model
        A classifier which will be used to combine the base estimators.
    """
    def __init__(self, models, final_model):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        final_model : model
            A classifier which will be used to combine the base estimators.
        """
        # parameters
        self.models = models
        self.final_model = final_model
    
    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The training data.
        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        dataset_copy = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            m.fit(dataset)
            dataset_copy.X = np.c_[dataset_copy.X, m.predict(dataset)]

        self.final_model.fit(dataset_copy)
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """
        dataset_copy = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            dataset_copy.X = np.c_[dataset_copy.X, m.predict(dataset)]

        return self.final_model.predict(dataset_copy)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression
    from si.io.csv import read_csv

    # load and split the dataset
    breast_dataset = read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\breast-bin.csv",
        features=True, label=True)
    dataset_ = breast_dataset
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

