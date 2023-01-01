import numpy as np

class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """

        Parameters
        ----------
        error
        learning_rate

        Returns
        -------

        """
        # Update the weight and bias
        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)

        # Error propagations return.
        return np.dot(error, self.weights.T)



class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        # attributes
        self.input_values = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        sigmoid_derivative = 1 / (1 + np.exp(-learning_rate))
        sigmoid_derivative = sigmoid_derivative * (1 - sigmoid_derivative)

        # Get error from previous layer
        return error * sigmoid_derivative


class SoftMaxActivation:
    """

    """
    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        """

        Parameters
        ----------
        input_data

        Returns
        -------

        """
        ez = np.exp(input_data)
        return ez / (np.sum(ez, keepdims=True))


class ReLUActivation:
    """

    """
    def __init__(self):
        self.data = None

    def forward(self, input_data: np.array):
        """

        Parameters
        ----------
        input_data:

        Returns
        -------

        """

        self.data = input_data
        return np.maximum(input_data, 0)

    def backward(self, error: np.ndarray, learning_rate: bool = 0.001):
        """

        Parameters
        ----------
        error:

        learning_rate:


        Returns
        -------

        """

        return error * np.where(learning_rate > 0, 1, 0)