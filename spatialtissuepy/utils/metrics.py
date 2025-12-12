"""
Common metrics and calculations for spatial analysis.
"""

import numpy as np
from typing import Union


def shannon_entropy(
    counts: Union[np.ndarray, list],
    normalize: bool = False
) -> float:
    """
    Calculate Shannon entropy of a distribution.

    Parameters
    ----------
    counts : array-like
        Counts or proportions for each category.
    normalize : bool, default False
        If True, normalize to [0, 1] range (divide by log(n_categories)).

    Returns
    -------
    float
        Shannon entropy value.

    Examples
    --------
    >>> shannon_entropy([10, 10, 10])  # uniform distribution
    1.0986...
    >>> shannon_entropy([30, 0, 0])    # single category
    0.0
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]  # Remove zeros
    
    if len(counts) == 0:
        return 0.0
    
    # Convert to proportions
    props = counts / counts.sum()
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(props * np.log(props))
    
    if normalize and len(counts) > 1:
        entropy = entropy / np.log(len(counts))
    
    return float(entropy)


def simpson_diversity(
    counts: Union[np.ndarray, list]
) -> float:
    """
    Calculate Simpson's diversity index.

    Also known as Gini-Simpson index. Represents probability that two
    randomly selected individuals belong to different categories.

    Parameters
    ----------
    counts : array-like
        Counts for each category.

    Returns
    -------
    float
        Simpson's diversity index (0 to 1).

    Examples
    --------
    >>> simpson_diversity([10, 10, 10])  # uniform
    0.666...
    >>> simpson_diversity([30, 0, 0])    # single category
    0.0
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    
    if total <= 1:
        return 0.0
    
    # D = 1 - sum(n_i * (n_i - 1)) / (N * (N - 1))
    numerator = np.sum(counts * (counts - 1))
    denominator = total * (total - 1)
    
    return float(1 - numerator / denominator)


def jaccard_index(
    set_a: Union[np.ndarray, set, list],
    set_b: Union[np.ndarray, set, list]
) -> float:
    """
    Calculate Jaccard similarity index between two sets.

    Parameters
    ----------
    set_a, set_b : array-like or set
        Two sets to compare.

    Returns
    -------
    float
        Jaccard index (0 to 1).

    Examples
    --------
    >>> jaccard_index([1, 2, 3], [2, 3, 4])
    0.5
    """
    set_a = set(set_a)
    set_b = set(set_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def euclidean_distance(
    point_a: np.ndarray,
    point_b: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    point_a, point_b : np.ndarray
        Coordinate arrays.

    Returns
    -------
    float
        Euclidean distance.
    """
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def normalize_counts(
    counts: np.ndarray,
    method: str = 'proportion'
) -> np.ndarray:
    """
    Normalize count data.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (n_samples, n_categories).
    method : str, default 'proportion'
        Normalization method:
        - 'proportion': Divide by row sum
        - 'zscore': Z-score normalization
        - 'minmax': Min-max scaling to [0, 1]

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    counts = np.asarray(counts, dtype=float)
    
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
    
    if method == 'proportion':
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return counts / row_sums
    
    elif method == 'zscore':
        mean = counts.mean(axis=0, keepdims=True)
        std = counts.std(axis=0, keepdims=True)
        std[std == 0] = 1
        return (counts - mean) / std
    
    elif method == 'minmax':
        min_val = counts.min(axis=0, keepdims=True)
        max_val = counts.max(axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (counts - min_val) / range_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
