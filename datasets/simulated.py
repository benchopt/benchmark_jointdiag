import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_matrices, n_features': [
            (40, 10)],  # slow to simulate big correlated design
    }

    def __init__(self, n_matrices=10, n_features=5, random_state=27):
        # Store the parameters of the dataset
        self.n_matrices = n_matrices
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        diagonals = rng.uniform(size=(self.n_matrices, self.n_features))
        A = rng.randn(self.n_features, self.n_features)  # mixing matrix
        C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset

        data = dict(C=C)

        return self.n_features, data
