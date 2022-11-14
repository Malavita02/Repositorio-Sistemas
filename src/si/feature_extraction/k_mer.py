import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
import itertools

class KMer:
    def __init__(self, k) -> None:
        self.k = k
        self.k_mers = None

    def fit(self, dataset):
        #pode fazer se com um dicionario e ir acrescentando 1 a 1
        # neste caso vamos usar o itertools.product para ter todas as combinações pociveis
        self.k_mers = ["".join(k_mer) for k_mer in itertools.product("ACTG", repeat= self.k)]
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
        sequences_k_mer_composition = [self._get_sequence_k_mers_composition(sequence) for sequence in dataset.X]
                                       

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":

    dataset = Dataset(X=np.array(["ATCG"],
                                  ["CTGA"],
                                  ["TGCA"]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"], label="y")
a = KMer(3)
a = a.fit_transform(dataset)
print(a.fit(dataset))
# ver git hub do fernando