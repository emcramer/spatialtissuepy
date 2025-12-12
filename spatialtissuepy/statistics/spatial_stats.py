"""
Classical spatial point pattern statistics.

This module implements Ripley's K-function and related statistics for analyzing
spatial point patterns. These functions test whether points are randomly distributed
(Complete Spatial Randomness, CSR), clustered, or dispersed.

Key Functions
-------------
- K-function: Cumulative neighbor count at distance r
- L-function: Variance-stabilized K (sqrt(K/π))
- H-function: Deviation from CSR (L - r)
- G-function: Nearest-neighbor distribution
- F-function: Empty-space distribution
- J-function: Ratio (1-G)/(1-F)
- Pair correlation function g(r): Local clustering intensity

References
----------
.. [1] Ripley, B. D. (1977). Modelling spatial patterns. Journal of the Royal
       Statistical Society: Series B (Methodological).
.. [2] Baddeley, A., Rubak, E., & Turner, R. (2015). Spatial Point Patterns:
       Methodology and Applications with R. CRC Press.
.. [3] Diggle, P. J. (2003). Statistical Analysis of Spatial Point Patterns.
       Arnold Publishers.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union, TYPE_CHECKING
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Ripley's K-function and variants
# -----------------------------------------------------------------------------

def ripleys_k(
    coordinates: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute Ripley's K-function for a point pattern.

    K(r) = (area / n²) * Σᵢ Σⱼ I(dᵢⱼ ≤ r) * wᵢⱼ

    where I is the indicator function and wᵢⱼ is the edge correction weight.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, 2).
    radii : np.ndarray
        Array of distances at which to evaluate K.
    area : float, optional
        Study region area. If None, uses bounding box area.
    edge_correction : str, default 'ripley'
        Edge correction method: 'none', 'ripley', or 'isotropic'.

    Returns
    -------
    np.ndarray
        K(r) values for each radius.

    Notes
    -----
    Under Complete Spatial Randomness (CSR), K(r) = π * r².
    - K(r) > π*r² indicates clustering
    - K(r) < π*r² indicates dispersion/regularity

    Examples
    --------
    >>> coords = np.random.rand(100, 2) * 1000
    >>> radii = np.linspace(0, 100, 20)
    >>> K = ripleys_k(coords, radii)
    >>> # Compare to CSR expectation
    >>> K_csr = np.pi * radii**2
    """
    n = len(coordinates)
    if n < 2:
        return np.zeros(len(radii))
    
    # Compute study region
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    
    if area is None:
        area = np.prod(max_coords - min_coords)
    
    if area <= 0:
        return np.zeros(len(radii))
    
    # Build KD-tree for efficient queries
    tree = cKDTree(coordinates)
    
    # Compute all pairwise distances up to max radius
    max_r = radii.max()
    pairs = tree.query_pairs(max_r, output_type='ndarray')
    
    if len(pairs) == 0:
        return np.zeros(len(radii))
    
    # Compute distances for pairs
    distances = np.linalg.norm(
        coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], axis=1
    )
    
    # Edge correction weights
    if edge_correction == 'none':
        weights = np.ones(len(distances))
    elif edge_correction == 'ripley':
        # Ripley's isotropic correction
        weights = _ripley_edge_correction(
            coordinates, pairs, distances, min_coords, max_coords
        )
    elif edge_correction == 'isotropic':
        # Same as ripley for rectangular regions
        weights = _ripley_edge_correction(
            coordinates, pairs, distances, min_coords, max_coords
        )
    else:
        raise ValueError(f"Unknown edge_correction: {edge_correction}")
    
    # Compute K for each radius
    K = np.zeros(len(radii))
    intensity = n / area
    
    for i, r in enumerate(radii):
        mask = distances <= r
        # Count pairs (multiply by 2 since we only have i<j pairs)
        K[i] = 2 * np.sum(weights[mask]) / (n * intensity)
    
    return K


def _ripley_edge_correction(
    coordinates: np.ndarray,
    pairs: np.ndarray,
    distances: np.ndarray,
    min_coords: np.ndarray,
    max_coords: np.ndarray
) -> np.ndarray:
    """
    Compute Ripley's isotropic edge correction weights.
    
    For each point, the weight accounts for the fraction of the circle
    that falls within the study region.
    """
    weights = np.ones(len(distances))
    
    for idx, (i, j) in enumerate(pairs):
        d = distances[idx]
        if d == 0:
            continue
        
        # For both points, compute fraction of circle in bounds
        w_i = _circle_in_rectangle_fraction(
            coordinates[i], d, min_coords, max_coords
        )
        w_j = _circle_in_rectangle_fraction(
            coordinates[j], d, min_coords, max_coords
        )
        
        # Average weight
        weights[idx] = 2.0 / (w_i + w_j) if (w_i + w_j) > 0 else 1.0
    
    return weights


def _circle_in_rectangle_fraction(
    center: np.ndarray,
    radius: float,
    min_coords: np.ndarray,
    max_coords: np.ndarray
) -> float:
    """
    Approximate fraction of circle circumference inside rectangle.
    Uses simple boundary distance approximation.
    """
    # Distance to each boundary
    d_left = center[0] - min_coords[0]
    d_right = max_coords[0] - center[0]
    d_bottom = center[1] - min_coords[1]
    d_top = max_coords[1] - center[1]
    
    # Count how many boundaries the circle crosses
    fraction = 1.0
    
    for d in [d_left, d_right, d_bottom, d_top]:
        if d < radius:
            # Approximate: fraction outside ≈ acos(d/r) / π
            if d >= 0:
                fraction -= np.arccos(min(d / radius, 1.0)) / (2 * np.pi)
            else:
                fraction -= 0.25  # Point outside boundary
    
    return max(fraction, 0.1)  # Minimum weight to avoid division issues


def ripleys_l(
    coordinates: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute Ripley's L-function (variance-stabilized K).

    L(r) = sqrt(K(r) / π)

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, 2).
    radii : np.ndarray
        Array of distances at which to evaluate L.
    area : float, optional
        Study region area.
    edge_correction : str, default 'ripley'
        Edge correction method.

    Returns
    -------
    np.ndarray
        L(r) values for each radius.

    Notes
    -----
    Under CSR, L(r) = r.
    - L(r) > r indicates clustering
    - L(r) < r indicates dispersion
    """
    K = ripleys_k(coordinates, radii, area, edge_correction)
    return np.sqrt(K / np.pi)


def ripleys_h(
    coordinates: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute Ripley's H-function (deviation from CSR).

    H(r) = L(r) - r = sqrt(K(r) / π) - r

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, 2).
    radii : np.ndarray
        Array of distances at which to evaluate H.
    area : float, optional
        Study region area.
    edge_correction : str, default 'ripley'
        Edge correction method.

    Returns
    -------
    np.ndarray
        H(r) values for each radius.

    Notes
    -----
    Under CSR, H(r) = 0.
    - H(r) > 0 indicates clustering at scale r
    - H(r) < 0 indicates dispersion at scale r
    
    This is the most interpretable form for detecting spatial patterns.
    """
    L = ripleys_l(coordinates, radii, area, edge_correction)
    return L - radii


# -----------------------------------------------------------------------------
# Cross-type K-function
# -----------------------------------------------------------------------------

def cross_k(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute cross-type K-function between two point patterns.

    Kab(r) = (area / (na * nb)) * Σᵢ Σⱼ I(dᵢⱼ ≤ r) * wᵢⱼ

    Parameters
    ----------
    coords_a : np.ndarray
        Coordinates of first point pattern.
    coords_b : np.ndarray
        Coordinates of second point pattern.
    radii : np.ndarray
        Array of distances at which to evaluate K.
    area : float, optional
        Study region area.
    edge_correction : str, default 'ripley'
        Edge correction method.

    Returns
    -------
    np.ndarray
        Kab(r) values for each radius.

    Notes
    -----
    Under spatial independence, Kab(r) = π * r².
    - Kab(r) > π*r² indicates attraction between types
    - Kab(r) < π*r² indicates repulsion/segregation
    """
    na, nb = len(coords_a), len(coords_b)
    
    if na == 0 or nb == 0:
        return np.zeros(len(radii))
    
    # Combined coordinates for area calculation
    all_coords = np.vstack([coords_a, coords_b])
    min_coords = all_coords.min(axis=0)
    max_coords = all_coords.max(axis=0)
    
    if area is None:
        area = np.prod(max_coords - min_coords)
    
    if area <= 0:
        return np.zeros(len(radii))
    
    # Compute cross-distances
    distances = cdist(coords_a, coords_b).ravel()
    
    # Edge correction (simplified for cross-K)
    if edge_correction == 'none':
        weights = np.ones(len(distances))
    else:
        # Use source point positions for edge correction
        weights = np.ones(len(distances))
        idx = 0
        for i in range(na):
            for j in range(nb):
                d = distances[idx]
                if d > 0:
                    w = _circle_in_rectangle_fraction(
                        coords_a[i], d, min_coords, max_coords
                    )
                    weights[idx] = 1.0 / max(w, 0.1)
                idx += 1
    
    # Compute K for each radius
    K = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        mask = distances <= r
        K[i] = area * np.sum(weights[mask]) / (na * nb)
    
    return K


def cross_l(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute cross-type L-function.

    Lab(r) = sqrt(Kab(r) / π)

    Parameters
    ----------
    coords_a, coords_b : np.ndarray
        Coordinates of two point patterns.
    radii : np.ndarray
        Evaluation distances.
    area : float, optional
        Study region area.
    edge_correction : str
        Edge correction method.

    Returns
    -------
    np.ndarray
        Lab(r) values.
    """
    K = cross_k(coords_a, coords_b, radii, area, edge_correction)
    return np.sqrt(K / np.pi)


def cross_h(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
    edge_correction: str = 'ripley'
) -> np.ndarray:
    """
    Compute cross-type H-function (deviation from independence).

    Hab(r) = Lab(r) - r

    Under independence, Hab(r) = 0.
    """
    L = cross_l(coords_a, coords_b, radii, area, edge_correction)
    return L - radii


# -----------------------------------------------------------------------------
# Nearest-neighbor G-function
# -----------------------------------------------------------------------------

def g_function(
    coordinates: np.ndarray,
    radii: np.ndarray,
    edge_correction: str = 'km'
) -> np.ndarray:
    """
    Compute the nearest-neighbor G-function.

    G(r) = P(nearest neighbor distance ≤ r)

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    radii : np.ndarray
        Evaluation distances.
    edge_correction : str, default 'km'
        Edge correction: 'none', 'km' (Kaplan-Meier), or 'rs' (reduced sample).

    Returns
    -------
    np.ndarray
        G(r) values (cumulative distribution).

    Notes
    -----
    Under CSR with intensity λ: G(r) = 1 - exp(-λ * π * r²)
    - G(r) above CSR indicates clustering
    - G(r) below CSR indicates regularity
    """
    n = len(coordinates)
    if n < 2:
        return np.zeros(len(radii))
    
    # Compute nearest neighbor distances
    tree = cKDTree(coordinates)
    nn_distances, _ = tree.query(coordinates, k=2)
    nn_distances = nn_distances[:, 1]  # Exclude self
    
    # Compute empirical CDF
    G = np.zeros(len(radii))
    
    if edge_correction == 'none':
        for i, r in enumerate(radii):
            G[i] = np.mean(nn_distances <= r)
    
    elif edge_correction in ['km', 'rs']:
        # Kaplan-Meier estimator with border distance censoring
        min_coords = coordinates.min(axis=0)
        max_coords = coordinates.max(axis=0)
        
        # Distance to nearest boundary
        border_dist = np.minimum(
            np.minimum(coordinates[:, 0] - min_coords[0], 
                      max_coords[0] - coordinates[:, 0]),
            np.minimum(coordinates[:, 1] - min_coords[1],
                      max_coords[1] - coordinates[:, 1])
        )
        
        for i, r in enumerate(radii):
            # Points with nn_distance <= r and not censored
            observed = (nn_distances <= r) & (border_dist >= nn_distances)
            # Points at risk (border_dist >= r or nn_distance <= r)
            at_risk = (border_dist >= r) | (nn_distances <= r)
            
            if np.sum(at_risk) > 0:
                G[i] = np.sum(observed) / np.sum(at_risk)
    
    else:
        raise ValueError(f"Unknown edge_correction: {edge_correction}")
    
    return G


def g_function_cross(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """
    Compute cross-type G-function (nearest type-b neighbor for type-a points).

    Gab(r) = P(distance from a-point to nearest b-point ≤ r)

    Parameters
    ----------
    coords_a : np.ndarray
        Source point coordinates.
    coords_b : np.ndarray
        Target point coordinates.
    radii : np.ndarray
        Evaluation distances.

    Returns
    -------
    np.ndarray
        Gab(r) values.
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.zeros(len(radii))
    
    # Nearest b-neighbor for each a-point
    tree_b = cKDTree(coords_b)
    nn_distances, _ = tree_b.query(coords_a, k=1)
    
    # Empirical CDF
    G = np.array([np.mean(nn_distances <= r) for r in radii])
    return G


# -----------------------------------------------------------------------------
# Empty-space F-function
# -----------------------------------------------------------------------------

def f_function(
    coordinates: np.ndarray,
    radii: np.ndarray,
    n_test_points: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Compute the empty-space F-function.

    F(r) = P(distance from random point to nearest data point ≤ r)

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    radii : np.ndarray
        Evaluation distances.
    n_test_points : int, default 1000
        Number of random test points.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        F(r) values.

    Notes
    -----
    Under CSR with intensity λ: F(r) = 1 - exp(-λ * π * r²)
    - F(r) above CSR indicates regularity (points spread out)
    - F(r) below CSR indicates clustering (large empty spaces)
    """
    if len(coordinates) == 0:
        return np.zeros(len(radii))
    
    rng = np.random.default_rng(seed)
    
    # Generate random test points in bounding box
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    
    test_points = rng.uniform(
        min_coords, max_coords, size=(n_test_points, coordinates.shape[1])
    )
    
    # Distance from test points to nearest data point
    tree = cKDTree(coordinates)
    nn_distances, _ = tree.query(test_points, k=1)
    
    # Empirical CDF
    F = np.array([np.mean(nn_distances <= r) for r in radii])
    return F


# -----------------------------------------------------------------------------
# J-function
# -----------------------------------------------------------------------------

def j_function(
    coordinates: np.ndarray,
    radii: np.ndarray,
    n_test_points: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Compute the J-function.

    J(r) = (1 - G(r)) / (1 - F(r))

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    radii : np.ndarray
        Evaluation distances.
    n_test_points : int, default 1000
        Number of test points for F-function.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        J(r) values.

    Notes
    -----
    Under CSR: J(r) = 1
    - J(r) < 1 indicates clustering
    - J(r) > 1 indicates regularity
    """
    G = g_function(coordinates, radii)
    F = f_function(coordinates, radii, n_test_points, seed)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        J = (1 - G) / (1 - F)
        J = np.where(np.isfinite(J), J, 1.0)
    
    return J


# -----------------------------------------------------------------------------
# Pair Correlation Function
# -----------------------------------------------------------------------------

def pair_correlation_function(
    coordinates: np.ndarray,
    radii: np.ndarray,
    bandwidth: Optional[float] = None,
    area: Optional[float] = None
) -> np.ndarray:
    """
    Compute the pair correlation function g(r).

    g(r) = K'(r) / (2πr)

    where K'(r) is the derivative of K(r).

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    radii : np.ndarray
        Evaluation distances.
    bandwidth : float, optional
        Kernel bandwidth for smoothing. Default: radii[1] - radii[0].
    area : float, optional
        Study region area.

    Returns
    -------
    np.ndarray
        g(r) values.

    Notes
    -----
    Under CSR: g(r) = 1 for all r.
    - g(r) > 1 indicates clustering at scale r
    - g(r) < 1 indicates inhibition at scale r
    """
    n = len(coordinates)
    if n < 2:
        return np.ones(len(radii))
    
    if bandwidth is None:
        if len(radii) > 1:
            bandwidth = (radii[-1] - radii[0]) / (len(radii) - 1)
        else:
            bandwidth = radii[0] / 10
    
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    
    if area is None:
        area = np.prod(max_coords - min_coords)
    
    if area <= 0:
        return np.ones(len(radii))
    
    intensity = n / area
    
    # Compute all pairwise distances
    tree = cKDTree(coordinates)
    max_r = radii.max() + 2 * bandwidth
    pairs = tree.query_pairs(max_r, output_type='ndarray')
    
    if len(pairs) == 0:
        return np.ones(len(radii))
    
    distances = np.linalg.norm(
        coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], axis=1
    )
    
    # Kernel density estimate at each radius
    g = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        if r <= 0:
            g[i] = 1.0
            continue
        
        # Epanechnikov kernel
        u = (distances - r) / bandwidth
        kernel_weights = np.where(
            np.abs(u) <= 1,
            0.75 * (1 - u**2) / bandwidth,
            0
        )
        
        # Expected count under CSR at distance r
        ring_area = 2 * np.pi * r * bandwidth
        expected = n * intensity * ring_area / 2
        
        if expected > 0:
            g[i] = 2 * np.sum(kernel_weights) / (n * intensity * 2 * np.pi * r)
        else:
            g[i] = 1.0
    
    return g


# -----------------------------------------------------------------------------
# CSR Envelope Testing
# -----------------------------------------------------------------------------

def csr_envelope(
    n_points: int,
    radii: np.ndarray,
    area: float,
    n_simulations: int = 99,
    statistic: str = 'K',
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate CSR simulation envelope for hypothesis testing.

    Parameters
    ----------
    n_points : int
        Number of points to simulate.
    radii : np.ndarray
        Evaluation distances.
    area : float
        Study region area.
    n_simulations : int, default 99
        Number of CSR simulations.
    statistic : str, default 'K'
        Statistic to compute: 'K', 'L', 'H', 'G', 'F', or 'g'.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'theoretical': Expected value under CSR
        - 'lower': Lower envelope (2.5th percentile)
        - 'upper': Upper envelope (97.5th percentile)
        - 'simulations': All simulated values (n_simulations, len(radii))

    Examples
    --------
    >>> envelope = csr_envelope(100, radii, area=1e6, statistic='H')
    >>> H_observed = ripleys_h(coords, radii)
    >>> # Points outside envelope indicate significant deviation from CSR
    >>> significant = (H_observed < envelope['lower']) | (H_observed > envelope['upper'])
    """
    rng = np.random.default_rng(seed)
    
    # Assume square region for simplicity
    side = np.sqrt(area)
    
    # Run simulations
    simulations = np.zeros((n_simulations, len(radii)))
    
    stat_func = {
        'K': lambda c: ripleys_k(c, radii, area),
        'L': lambda c: ripleys_l(c, radii, area),
        'H': lambda c: ripleys_h(c, radii, area),
        'G': lambda c: g_function(c, radii),
        'F': lambda c: f_function(c, radii, seed=None),
        'g': lambda c: pair_correlation_function(c, radii, area=area),
    }.get(statistic)
    
    if stat_func is None:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    for i in range(n_simulations):
        # Generate CSR pattern
        coords = rng.uniform(0, side, size=(n_points, 2))
        simulations[i] = stat_func(coords)
    
    # Compute envelopes
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)
    
    # Theoretical values under CSR
    intensity = n_points / area
    if statistic == 'K':
        theoretical = np.pi * radii**2
    elif statistic == 'L':
        theoretical = radii
    elif statistic == 'H':
        theoretical = np.zeros(len(radii))
    elif statistic in ['G', 'F']:
        theoretical = 1 - np.exp(-intensity * np.pi * radii**2)
    elif statistic == 'g':
        theoretical = np.ones(len(radii))
    else:
        theoretical = np.mean(simulations, axis=0)
    
    return {
        'theoretical': theoretical,
        'lower': lower,
        'upper': upper,
        'mean': np.mean(simulations, axis=0),
        'simulations': simulations,
    }


# -----------------------------------------------------------------------------
# High-level functions for SpatialTissueData
# -----------------------------------------------------------------------------

def spatial_statistics(
    data: 'SpatialTissueData',
    radii: Optional[np.ndarray] = None,
    n_radii: int = 50,
    max_radius: Optional[float] = None,
    statistics: List[str] = ['K', 'L', 'H'],
    cell_type: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compute multiple spatial statistics for tissue data.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radii : np.ndarray, optional
        Evaluation distances. If None, auto-generated.
    n_radii : int, default 50
        Number of radii if auto-generating.
    max_radius : float, optional
        Maximum radius. Default: 1/4 of smallest extent.
    statistics : list of str, default ['K', 'L', 'H']
        Statistics to compute.
    cell_type : str, optional
        Compute for specific cell type only.

    Returns
    -------
    dict
        Dictionary with 'radii' and computed statistics.
    """
    # Select coordinates
    if cell_type is not None:
        idx = data.get_cells_by_type(cell_type)
        coords = data._coordinates[idx, :2]
    else:
        coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        raise ValueError("Need at least 2 points for spatial statistics")
    
    # Auto-generate radii
    if radii is None:
        if max_radius is None:
            extent = data.extent
            max_radius = min(extent['x'], extent['y']) / 4
        radii = np.linspace(0, max_radius, n_radii)
    
    # Compute area
    bounds = data.bounds
    area = (bounds['x'][1] - bounds['x'][0]) * (bounds['y'][1] - bounds['y'][0])
    
    result = {'radii': radii}
    
    for stat in statistics:
        if stat == 'K':
            result['K'] = ripleys_k(coords, radii, area)
        elif stat == 'L':
            result['L'] = ripleys_l(coords, radii, area)
        elif stat == 'H':
            result['H'] = ripleys_h(coords, radii, area)
        elif stat == 'G':
            result['G'] = g_function(coords, radii)
        elif stat == 'F':
            result['F'] = f_function(coords, radii)
        elif stat == 'J':
            result['J'] = j_function(coords, radii)
        elif stat == 'g':
            result['g'] = pair_correlation_function(coords, radii, area=area)
    
    return result


def cross_type_statistics(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radii: Optional[np.ndarray] = None,
    n_radii: int = 50,
    max_radius: Optional[float] = None,
    statistics: List[str] = ['K', 'L', 'H']
) -> Dict[str, np.ndarray]:
    """
    Compute cross-type spatial statistics between two cell types.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    type_a, type_b : str
        Cell types to analyze.
    radii : np.ndarray, optional
        Evaluation distances.
    n_radii : int, default 50
        Number of radii.
    max_radius : float, optional
        Maximum radius.
    statistics : list of str
        Statistics to compute.

    Returns
    -------
    dict
        Dictionary with 'radii' and cross-type statistics.
    """
    coords_a = data._coordinates[data.get_cells_by_type(type_a), :2]
    coords_b = data._coordinates[data.get_cells_by_type(type_b), :2]
    
    if len(coords_a) == 0 or len(coords_b) == 0:
        raise ValueError(f"Need points of both types: {type_a}, {type_b}")
    
    # Auto-generate radii
    if radii is None:
        if max_radius is None:
            extent = data.extent
            max_radius = min(extent['x'], extent['y']) / 4
        radii = np.linspace(0, max_radius, n_radii)
    
    bounds = data.bounds
    area = (bounds['x'][1] - bounds['x'][0]) * (bounds['y'][1] - bounds['y'][0])
    
    result = {'radii': radii}
    
    for stat in statistics:
        if stat == 'K':
            result['K'] = cross_k(coords_a, coords_b, radii, area)
        elif stat == 'L':
            result['L'] = cross_l(coords_a, coords_b, radii, area)
        elif stat == 'H':
            result['H'] = cross_h(coords_a, coords_b, radii, area)
        elif stat == 'G':
            result['G'] = g_function_cross(coords_a, coords_b, radii)
    
    return result
