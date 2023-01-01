import numpy as np
from src.si.data.dataset import Dataset

class PCA:
    """
    Linear algebra technique that reduces dataset size.
    This PCA is implemented using SVA - Singular Value Decomposition.

    Parameters
    ----------
    n_components: int
        Numbeer of components

    Attributes
    ----------
    mean: float
        Mean of the samples
    components: list
        Principal components aka unitary matrix of eigenvectors
    explained_variance: float
        Explained varience aka diagonal matrix of eigenvectors

    """
    def __init__(self, n_components: int = 2):
        """
        PCA using Singular Value Decomposition (SVA).

        Parameters
        ----------
        n_components: int
            Number of components
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, dataset: Dataset) -> 'PCA':
        """
        It fits PCA reducing the dataset.
        The PCA algorithm starts by centering the data using the mean value.
        Gets the components and the explained variance using numpy.linalg.svd().

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        PCA
            PCA object.
        """
        self.mean = np.mean(dataset.X, axis=0)
        self.centred_data = dataset.X - self.mean
        
        U, S, Vt = np.linalg.svd(dataset.X, full_matrices=False)
        self.components = Vt[:self.n_components]
        n = len(dataset.X)
        EV = S**2/(n-1)
        self.explained_variance = EV[:self.n_components]

        return self
    
    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset.
        It computes de dot product of the dataset with the reduced dataset

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Tranformed dataset.
        """
        U, S, Vt = np.linalg.svd(dataset.X, full_matrices=False)
        V = np.transpose(Vt)
        X_reduced = np.dot(dataset.X, V)

        return X_reduced

    def fit_transform(self,dataset: Dataset) -> np.ndarray:
        """
         It fits and transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Tranformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":

    # test using iris.csv dataset
    from src.si.io.csv import read_csv
    dataset = read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\iris.csv", features=True, label= True)
    a = PCA(n_components = 2)
    print(a.fit_transform(dataset=dataset))