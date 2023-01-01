import numpy as np

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid function of the inputted values.

    Parameters
    ----------
    x: np.ndarray
        Input values
    Returns
    -------
    np.ndarray: Sigmoid function value.
    """
    return 1/(1 + np.exp(-x))