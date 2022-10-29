from random import seed
import numpy as np

def train_test_split(dataset, test_size, random_state = 22):
    numsts = int(dataset.get_shape()[0] * test_size)
    indices = np.random.permutation(dataset.get_shape()[0], seed = random_state)
    train = dataset.X[indices[:-numsts]]
    test = dataset.X[indices[-numsts:]]
    return (train, test)

# Falta testar mas em principio vai dar