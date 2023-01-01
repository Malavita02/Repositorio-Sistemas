import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the cross entropy of the model on the given dataset
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    cross_entropy: float
        The cross entropy of the model
    """
    return -np.sum(y_true*np.log(y_pred))/len(y_true)



def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the derivative of the cross entropy of the model on the given dataset
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    cross_entropy_derivative: float
        The derivative cross entropy of the model
    """
    return -y_true / (len(y_true)*y_pred)