"""
Spatial sampling methods for Spatial LDA.

This module provides various sampling strategies for selecting index cells
when fitting Spatial LDA models. Sampling can improve computational efficiency
and reduce redundancy when working with densely sampled tissues.

Key Methods
-----------
- Poisson disk sampling: Ensures minimum distance between samples
- Grid sampling: Regular grid with optional jitter
- Random sampling: Uniform random selection
- Stratified sampling: Proportional sampling by cell type
"""

from __future__ import annotations
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


def poisson_disk_sample(
    data: 'SpatialTissueData',
    min_distance: float,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample cells using Poisson disk sampling.
    
    This ensures a minimum distance between sampled cells, providing
    even spatial coverage while avoiding redundant nearby samples.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    min_distance : float
        Minimum distance between sampled cells.
    max_samples : int, optional
        Maximum number of samples. If None, sample as many as possible.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    np.ndarray
        Indices of sampled cells.
        
    Notes
    -----
    Uses a greedy dart-throwing algorithm:
    1. Randomly shuffle cells
    2. For each cell, accept if at least min_distance from all accepted
    
    This provides approximately uniform spatial coverage.
    
    Examples
    --------
    >>> # Sample cells at least 100 µm apart
    >>> indices = poisson_disk_sample(data, min_distance=100)
    >>> print(f"Sampled {len(indices)} cells")
    """
    rng = np.random.default_rng(seed)
    
    coords = data._coordinates
    n_cells = len(coords)
    
    # Shuffle order for random selection
    order = rng.permutation(n_cells)
    
    # Accepted samples
    accepted = []
    accepted_coords = []
    
    # Build KD-tree incrementally for efficiency
    tree = None
    
    for idx in order:
        if max_samples is not None and len(accepted) >= max_samples:
            break
        
        coord = coords[idx]
        
        # Check distance to all accepted
        if len(accepted) == 0:
            # First sample always accepted
            accepted.append(idx)
            accepted_coords.append(coord)
            tree = cKDTree([coord])
        else:
            # Check if far enough from all accepted
            dist, _ = tree.query(coord)
            
            if dist >= min_distance:
                accepted.append(idx)
                accepted_coords.append(coord)
                # Rebuild tree (could be optimized)
                tree = cKDTree(accepted_coords)
    
    return np.array(accepted)


def grid_sample(
    data: 'SpatialTissueData',
    spacing: float = 50.0,
    jitter: float = 0.0,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample cells on a regular grid.
    
    Selects cells closest to regular grid points.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    spacing : float
        Grid spacing (distance between grid points).
    jitter : float, default 0.0
        Random jitter to add to grid points (as fraction of spacing).
    seed : int, optional
        Random seed for jitter.
    **kwargs
        Additional arguments, including grid_size (alias for spacing).
        
    Returns
    -------
    np.ndarray
        Indices of sampled cells.
        
    Examples
    --------
    >>> indices = grid_sample(data, spacing=50, jitter=0.1)
    """
    rng = np.random.default_rng(seed)
    
    if 'grid_size' in kwargs:
        spacing = kwargs.pop('grid_size')
    
    coords = data._coordinates
    bounds = data.bounds
    
    # Determine dimensionality
    n_dims = 2 if coords.shape[1] <= 2 else 3
    if coords.shape[1] >= 3 and np.std(coords[:, 2]) < 1e-6:
        n_dims = 2  # Effectively 2D
    
    # Create grid points
    if n_dims == 2:
        x_grid = np.arange(bounds['x'][0], bounds['x'][1] + spacing, spacing)
        y_grid = np.arange(bounds['y'][0], bounds['y'][1] + spacing, spacing)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        x_grid = np.arange(bounds['x'][0], bounds['x'][1] + spacing, spacing)
        y_grid = np.arange(bounds['y'][0], bounds['y'][1] + spacing, spacing)
        z_grid = np.arange(bounds['z'][0], bounds['z'][1] + spacing, spacing)
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Add jitter
    if jitter > 0:
        noise = rng.uniform(-jitter * spacing, jitter * spacing, grid_points.shape)
        grid_points = grid_points + noise
    
    # Find nearest cell to each grid point
    tree = cKDTree(coords[:, :n_dims])
    _, indices = tree.query(grid_points[:, :n_dims])
    
    # Remove duplicates (multiple grid points may map to same cell)
    unique_indices = np.unique(indices)
    
    return unique_indices


def random_sample(
    data: 'SpatialTissueData',
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Uniform random sampling of cells.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_samples : int
        Number of samples to select.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    np.ndarray
        Indices of sampled cells.
    """
    rng = np.random.default_rng(seed)
    
    n_cells = data.n_cells
    n_samples = min(n_samples, n_cells)
    
    return rng.choice(n_cells, size=n_samples, replace=False)


def stratified_sample(
    data: 'SpatialTissueData',
    n_samples: int = 100,
    by: str = 'cell_type',
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Stratified sampling to maintain cell type proportions.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_samples : int
        Total number of samples to select.
    by : str, default 'cell_type'
        Stratification variable (currently only 'cell_type' supported).
    seed : int, optional
        Random seed.
    **kwargs
        Additional arguments, including n_per_type (multiplies by n_types).
        
    Returns
    -------
    np.ndarray
        Indices of sampled cells.
        
    Notes
    -----
    Samples proportionally from each cell type to maintain the
    original composition in the sample.
    """
    rng = np.random.default_rng(seed)
    
    if 'n_per_type' in kwargs:
        n_samples = kwargs.pop('n_per_type') * len(data.cell_types_unique)
    
    cell_types = data._cell_types
    unique_types = data.cell_types_unique
    
    n_cells = data.n_cells
    n_samples = min(n_samples, n_cells)
    
    # Calculate samples per type
    type_counts = {
        ct: np.sum(cell_types == ct) for ct in unique_types
    }
    
    samples_per_type = {
        ct: max(1, int(np.round(n_samples * count / n_cells)))
        for ct, count in type_counts.items()
    }
    
    # Adjust to match total
    total = sum(samples_per_type.values())
    if total > n_samples:
        # Reduce from largest groups
        sorted_types = sorted(samples_per_type.keys(), 
                             key=lambda x: samples_per_type[x], reverse=True)
        for ct in sorted_types:
            if total <= n_samples:
                break
            if samples_per_type[ct] > 1:
                samples_per_type[ct] -= 1
                total -= 1
    
    # Sample from each type
    all_indices = []
    
    for ct in unique_types:
        type_indices = np.where(cell_types == ct)[0]
        n_to_sample = min(samples_per_type[ct], len(type_indices))
        
        sampled = rng.choice(type_indices, size=n_to_sample, replace=False)
        all_indices.extend(sampled)
    
    return np.array(all_indices)


def spatial_stratified_sample(
    data: 'SpatialTissueData',
    n_samples: int,
    n_regions: int = 4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Stratified sampling by spatial region.
    
    Divides the tissue into spatial regions and samples proportionally
    from each region to ensure spatial coverage.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_samples : int
        Total number of samples.
    n_regions : int, default 4
        Number of spatial regions (will create n_regions x n_regions grid).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    np.ndarray
        Indices of sampled cells.
    """
    rng = np.random.default_rng(seed)
    
    coords = data._coordinates
    bounds = data.bounds
    
    # Create region grid
    x_edges = np.linspace(bounds['x'][0], bounds['x'][1], n_regions + 1)
    y_edges = np.linspace(bounds['y'][0], bounds['y'][1], n_regions + 1)
    
    # Assign cells to regions
    x_bins = np.digitize(coords[:, 0], x_edges) - 1
    y_bins = np.digitize(coords[:, 1], y_edges) - 1
    
    # Clip to valid range
    x_bins = np.clip(x_bins, 0, n_regions - 1)
    y_bins = np.clip(y_bins, 0, n_regions - 1)
    
    region_ids = x_bins * n_regions + y_bins
    
    # Sample from each region
    samples_per_region = max(1, n_samples // (n_regions * n_regions))
    
    all_indices = []
    
    for region_id in range(n_regions * n_regions):
        region_cells = np.where(region_ids == region_id)[0]
        
        if len(region_cells) > 0:
            n_to_sample = min(samples_per_region, len(region_cells))
            sampled = rng.choice(region_cells, size=n_to_sample, replace=False)
            all_indices.extend(sampled)
    
    # If we need more samples, randomly add from all cells
    if len(all_indices) < n_samples:
        remaining = set(range(data.n_cells)) - set(all_indices)
        if remaining:
            n_extra = min(n_samples - len(all_indices), len(remaining))
            extra = rng.choice(list(remaining), size=n_extra, replace=False)
            all_indices.extend(extra)
    
    return np.array(all_indices[:n_samples])
