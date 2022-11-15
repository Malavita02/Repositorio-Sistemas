import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def train_test_split(dataset, test_size, random_state = 22):
    # set random state
    np.random.seed(random_state)

    # get dataset size
    n_samples = dataset.get_shape()[0]

    # get number of samples in the test set
    n_test = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get samples in the test set
    test_idxs = permutations[:n_test]

    # get samples in the training set
    train_idxs = permutations[n_test:]

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

# Falta testar mas em principio vai dar