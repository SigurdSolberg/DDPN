from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import tqdm
import multiprocessing
import itertools
import gudhi
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

class DistributedHomology():
    """
    Computes the distributed homology of a dataset using either the Alpha Complex or Rips Complex.

    Parameters
    ----------
    max_pd_size : int, optional
        The maximum size of the persistence diagrams after padding with zeros. Default is 0.
    """

    def __init__(self) -> None:
        self.max_pd_size = 0
        self.subsets = []

    @timer
    def fit(self, X, k, m, alpha=True, normalization = [], max_featues = 1000000000):
        """
        Computes the distributed homology of a dataset.

        Parameters
        ----------
        X : numpy.ndarray
            The dataset as an array of shape (n_samples, n_features).
        k : int
            The size of the subsets.
        m : int
            The number of subsets.
        alpha : bool, optional
            Whether to use the Alpha Complex instead of the Rips Complex. Default is False.
        normalization : list, optional
            A list of normalization methods to apply to the point clouds. Default is [].
        max_featues : int, optional
            The maximum number of features to include in the persistence diagrams for each dimension. Default is 1000000000.

        Returns
        -------
        np.array
           An array of shape (n_samples, n_subsets, n_features, 3) containing the persistence diagrams of each subset of all point clouds.
           Each feature is represented by a tuple (birth, death, dimension).

        Notes
        -----
            - The persistence diagrams are padded with 0s to make them all the same size.
            - All subsets are saved in self.subsets.
        """

        self.subsets = []
        data = []

        # Compute DH for each pointcloud
        for pointcloud in tqdm(X, desc="Processing", total=len(X)):

            # Normalize pointcloud
            for normalization_method in normalization:
                pointcloud = normalization_method(pointcloud)

            if m > 1000:
                cpu = multiprocessing.cpu_count()
                with multiprocessing.Pool(cpu) as p:
                    a = p.starmap(self.compute_distributed_homology, [[pointcloud, k, m // cpu, alpha, max_featues], ] * cpu)
                data.append(list(itertools.chain.from_iterable(a)))
            else:
                data.append(self.compute_distributed_homology(pointcloud, k, m, alpha, max_featues))

        # Pad diagrams with 0s to make them all the same size
        for i, pointcloud in enumerate(data):
            for j, diagram in enumerate(pointcloud):
                data[i][j] = np.pad(diagram, ((0, self.max_pd_size - len(diagram)), (0, 0)), 'constant', constant_values=0)

        self.subsets = np.array(self.subsets)
        self.compute_distance_matrices()
        return np.array(data)

    def compute_distributed_homology(self, X, k, m, alpha=True, max_featues = 1000000000):
        """
        Computes the distributed homology of a single point cloud.

        Parameters
        ----------
        pointcloud : numpy.ndarray
            The point cloud as an array of shape (n_samples, n_features).
        k : int
            The size of the subsets.
        m : int
            The number of subsets.
        max_featues : int
            The maximum number of features to include in the persistence diagrams for each dimension.
        alpha : bool
            Whether to use the Alpha Complex instead of the Rips Complex.
        show : bool
            Whether to plot the persistence diagrams.

        Returns
        -------
        List
            A list of persistence diagrams, where each diagram is a list of arrays of (birth, death) pairs for each dimension.
        """

        subsets = _get_subsets(X, k, m)
        dh = []

        for subset in subsets:
            filtered_complex = gudhi.AlphaComplex(subset) if alpha else gudhi.RipsComplex(points = subset)

            # Calculate Persistent Homology of subset
            simplex_tree = filtered_complex.create_simplex_tree()
            simplex_tree.compute_persistence()

            pd = []
            dims = len(simplex_tree.betti_numbers())
            for dim in range(dims):

                dim_features = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(dim_features) > 0:

                    # Add death-birth column
                    dim_features = np.c_[dim_features, dim_features[:, 1] - dim_features[:, 0]]

                    # Sort by death-birth column, and select max_featues most persistent features
                    if max_featues != 1000000000:
                        dim_features = dim_features[np.argsort(dim_features[:, 2])][-max_featues:]

                    # Add one-hot encoding of dimension
                    ohe = np.zeros(shape = (len(dim_features), dims))
                    ohe[:, dim] = 1
                    dim_features = np.c_[dim_features, ohe]

                    pd.append(dim_features)

            pd[0] = np.delete(pd[0], -1, axis = 0)
            
            pd = np.concatenate(pd)
            dh.append(pd)

            if len(pd) > self.max_pd_size:
                self.max_pd_size = len(pd)

        self.subsets.append(subsets)
        return dh
    
    def compute_distance_matrices(self):
        """
        Computes the distance matrices of distances between all subsets of a point cloud.

        [NOT IMPLEMENTED]
        """
        pass

#@numba.njit
def _get_subsets(X, k, m):
    """
    Uniformly samples m subsets of size k from X.
    If m > len(X), subsets are sampled with replacement.

    Args:
        X (numpy.array):    Set of n datapoints - n x d
        k (int):            Size of subsets
        m (int):            Number of subsets

    Returns:
        numpy.array:        m subsets from X of size k - m x k x d
    """
    subsets = np.zeros(shape=(m, k, X.shape[-1]))
    for i in range(m):
        if len(X) < k:
            subsets[i] = X[np.random.choice(len(X), k, replace=True)]
        else:
            subsets[i] = X[np.random.choice(len(X), k, replace=False)]
    return subsets
