import numpy as np

def euclidean_distance(x, y):
    distance = np.sqrt(((x - y) ** 2).sum(axis=1))
    return distance
