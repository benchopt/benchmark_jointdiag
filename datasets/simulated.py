from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_matrices, n_features, noise_level': [
            (100, 5, 0.01), (100, 5, 0.1)]
    }

    def __init__(self, n_matrices=10, n_features=5, noise_level=0.001,
                 random_state=27):
        self.n_matrices = n_matrices
        self.n_features = n_features
        self.noise_level = noise_level
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        diagonals = rng.uniform(size=(self.n_matrices, self.n_features))
        A = rng.randn(self.n_features, self.n_features)  # mixing matrix
        C = (A[None, :, :] * diagonals[:, None, :]) @ A.T  # clean dataset
        N = self.noise_level * rng.randn(self.n_matrices, self.n_features,
                                         self.n_features)
        # create random psd matrices and add them to C
        N = N @ np.transpose(N, axes=(0, 2, 1))
        C += N
        return dict(C=C, A=A)
