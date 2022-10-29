import numpy as np

def accuracy(y_true, y_pred):
    pecc = np.sum(y_true == y_pred) / len(y_true)
    return pecc

# Falta testar mas em principio vai dar