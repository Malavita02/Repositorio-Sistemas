import numpy as np
from src.si.data.dataset import Dataset
from typing import Callable
from src.si.metrics.mse import mse, mse_derivative
from src.si.metrics.accuracy import accuracy

class NN:
    def __init__(self, layers: list, epochs: int = 1000, learning_rate: float = 0.01, loss: Callable = mse, loss_derivative: Callable = mse_derivative, verbose: bool = True):
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        # attributes
        self.history = {}


    def fit(self, dataset: Dataset) -> "NN":

        for epoch in range(self.epochs):
            y_pred = np.array(dataset.X)
            y_true = np.reshape(dataset.y, (-1, 1))

            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            error = self.loss_derivative(y_true, y_pred)
            for layer in self.layers[::-1]:  # backwards propagation
                error = layer.backward(error, self.learning_rate)

            # save history
            cost = self.loss(y_true, y_pred)
            self.history[epoch] = cost

            if self.verbose:
                print(f'Epoch {epoch}')

            return self

    def predict(self, dataset: Dataset):
        pred = dataset.X
        for layer in self.layers:
            pred = layer.forward(pred)

        return pred

    def cost(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)

if __name__ == '__main__':
    from si.neural_networks.layer import Dense, SigmoidActivation, SoftMaxActivation

    X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,2]])

    Y = np.array([1,
                0,
                0,
                1])

    dataset = Dataset(X, Y, ['x1', 'x2'], 'x1 XNOR x2')
    print(dataset.to_dataframe())


    w1 = np.array([[20, -20],
                    [20, -20]])

    b1 = np.array([[-30, 10]])


    l1 = Dense(input_size = 2, output_size=2)
    l1.weights = w1
    l1.bias = b1


    w2 = np.array([[20],
                    [20]])

    b2 = np.array([[-10]])

    l2 = Dense(input_size = 2, output_size=1)
    l2.weights = w2
    l2.bias = b2

    l1_sa = SigmoidActivation()
    l2_sa = SigmoidActivation()

    nn_model_sa = NN(layers=[l1, l1_sa, l2, l2_sa])
    nn_model_sa.fit(dataset=dataset)


    print(nn_model_sa.predict(dataset))





