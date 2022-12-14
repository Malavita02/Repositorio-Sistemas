import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
import itertools
from sklearn.preprocessing import StandardScaler


class KMer:
    def __init__(self, k, alphabet = "ACTG") -> None:
        self.k = k
        self.k_mers = None
        self.alphabet = alphabet

    def fit(self, dataset):
        #pode fazer se com um dicionario e ir acrescentando 1 a 1
        # neste caso vamos usar o itertools.product para ter todas as combinações pociveis
        self.k_mers = ["".join(k_mer) for k_mer in itertools.product(self.alphabet, repeat= self.k)]
        return self
    
    def _get_sequence_k_mers_composition(self, sequence):
        k_mers = {}
        for mers in self.k_mers:
            k_mers[mers] = 0
        for i in range(len(sequence)- self.k +1):
            k_mer = sequence[i:i+self.k]
            k_mers[k_mer] += 1
        #normalize the counts
        return np.array([k_mers[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset):
        sequences_k_mer_composition = [self._get_sequence_k_mers_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)    

        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)                          

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    """dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')
    k_mer_ = KMer(k=3)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)"""
    from model_selection.split import train_test_split
    from linear_model.logistic_regression import LogisticRegression

    data = Dataset()
    tranposter = data.new_read_csv(filename= r'/home/tiago/AulasSegundoAno/Repositorio-Sistemas/datasets/transporters.csv', sep = ",", features= "Sequence", label="label")
    tranposter.X = StandardScaler().fit_transform(tranposter.X)
    dataset_train, dataset_test = train_test_split(tranposter, test_size=0.2)
    log_reg = LogisticRegression()
    log_reg.fit(dataset_train)
    score = log_reg.score(dataset_test)
    print(score)



