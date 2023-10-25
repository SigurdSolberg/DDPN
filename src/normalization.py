import numpy as np
from scipy.stats import norm

def center(point_cloud):
    """
    Centers the point cloud around the origin.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        The point cloud as an array of shape (n_samples, n_features).

    Note
    ----
    We dont need to do this when computing persistence, but it is useful for visualization 
    and other set of set datasets.
    """
    centroid = np.mean(point_cloud, axis=0)
    return point_cloud - centroid

def normalize_size(point_cloud, radius = 4):
    """
    Scales the point cloud to fit within a unit sphere. 

    Parameters
    ----------
    point_cloud : numpy.ndarray
        The point cloud as an array of shape (n_samples, n_features).
    radius : float
        The radius of the sphere.

    Note
    ----
    We actually want to scale the magnitude of the distances (birth, death of topological features), thus
    setting radius to 1 is not necessarily the best choice.
    """
    max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
    return point_cloud * radius/ max_distance

def normalize_distances(point_cloud):
    """
    Scales the point cloud so that the maximum distance is 1.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        The point cloud as an array of shape (n_samples, n_features).
    """
    max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
    return point_cloud / max_distance

def normalize_size_dwise(point_cloud):
    """
    Scales the point cloud so that the maximum distance in each dimension is 1.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        The point cloud as an array of shape (n_samples, n_features).
    """
    max_distances = np.max(point_cloud, axis=0)
    return point_cloud / max_distances