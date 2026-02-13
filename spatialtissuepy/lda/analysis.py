"""
Analysis functions for Spatial LDA results.

This module provides functions for interpreting and analyzing
fitted Spatial LDA models and their topic assignments.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    from .spatial_lda import SpatialLDA


# -----------------------------------------------------------------------------
# Topic Characterization
# -----------------------------------------------------------------------------

def topic_cell_type_matrix(
    model: 'SpatialLDA',
    normalize: bool = True
) -> pd.DataFrame:
    """
    Get the topic-cell type association matrix.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    normalize : bool, default True
        If True, rows sum to 1 (probability distribution).
        
    Returns
    -------
    pd.DataFrame
        Matrix with topics as rows, cell types as columns.
    """
    if not model._is_fitted:
        raise RuntimeError("Model not fitted.")
    
    matrix = model.topic_cell_type_matrix_.copy()
    
    if not normalize:
        # Return unnormalized (LDA components)
        matrix = model._lda_model.components_.copy()
    
    return pd.DataFrame(
        matrix,
        index=[f'Topic_{i}' for i in range(model.n_topics)],
        columns=model.cell_types_
    )


def topic_enrichment(
    model: 'SpatialLDA',
    baseline: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute cell type enrichment in each topic relative to baseline.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    baseline : np.ndarray, optional
        Baseline cell type proportions. If None, uses uniform distribution.
        
    Returns
    -------
    pd.DataFrame
        Log2 enrichment scores (positive = enriched, negative = depleted).
    """
    if not model._is_fitted:
        raise RuntimeError("Model not fitted.")
    
    topic_matrix = model.topic_cell_type_matrix_
    n_types = topic_matrix.shape[1]
    
    if baseline is None:
        baseline = np.ones(n_types) / n_types
    
    # Compute log2 fold change
    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = np.log2(topic_matrix / baseline)
        enrichment = np.nan_to_num(enrichment, nan=0, posinf=5, neginf=-5)
    
    return pd.DataFrame(
        enrichment,
        index=[f'Topic_{i}' for i in range(model.n_topics)],
        columns=model.cell_types_
    )


def dominant_topic_per_cell(
    model: Union['SpatialLDA', np.ndarray],
    data: Optional['SpatialTissueData'] = None,
    return_weights: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get the dominant (most probable) topic for each cell.
    
    Parameters
    ----------
    model : SpatialLDA or np.ndarray
        Fitted model OR precomputed topic weight matrix.
    data : SpatialTissueData, optional
        Data to analyze (required if model is SpatialLDA).
    return_weights : bool, default False
        If True, also return the weight of the dominant topic.
        
    Returns
    -------
    np.ndarray
        Dominant topic indices (n_cells,).
    weights : np.ndarray, optional
        Weight of dominant topic (if return_weights=True).
    """
    if isinstance(model, np.ndarray):
        topic_weights = model
    else:
        if data is None:
            raise ValueError("data required when providing SpatialLDA model")
        topic_weights = model.transform(data)
    
    dominant = np.argmax(topic_weights, axis=1)
    
    if return_weights:
        weights = np.max(topic_weights, axis=1)
        return dominant, weights
    
    return dominant


def topic_assignment_uncertainty(
    model: Union['SpatialLDA', np.ndarray],
    data: Optional['SpatialTissueData'] = None
) -> np.ndarray:
    """
    Compute entropy of topic assignments as uncertainty measure.
    
    Higher entropy = more uncertain assignment. Ranges from 0-1.
    
    Parameters
    ----------
    model : SpatialLDA or np.ndarray
        Fitted model OR precomputed topic weight matrix.
    data : SpatialTissueData, optional
        Data to analyze (required if model is SpatialLDA).
        
    Returns
    -------
    np.ndarray
        Entropy values for each cell.
    """
    if isinstance(model, np.ndarray):
        topic_weights = model
    else:
        if data is None:
            raise ValueError("data required when providing SpatialLDA model")
        topic_weights = model.transform(data)
    
    # Compute entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        log_weights = np.log2(topic_weights + 1e-10)
        entropy = -np.sum(topic_weights * log_weights, axis=1)
    
    return entropy


# -----------------------------------------------------------------------------
# Spatial Analysis of Topics
# -----------------------------------------------------------------------------

def topic_spatial_distribution(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    topic_idx: int
) -> Dict[str, np.ndarray]:
    """
    Analyze the spatial distribution of a topic.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
    topic_idx : int
        Index of topic to analyze.
        
    Returns
    -------
    dict
        Dictionary with:
        - 'positions': Coordinates of cells
        - 'weights': Topic weight for each cell
        - 'centroid': Weighted centroid of topic
        - 'spread': Weighted standard deviation
    """
    topic_weights = model.transform(data)
    weights = topic_weights[:, topic_idx]
    coords = data._coordinates
    
    # Weighted centroid
    total_weight = np.sum(weights)
    if total_weight > 0:
        centroid = np.average(coords, weights=weights, axis=0)
    else:
        centroid = np.mean(coords, axis=0)
    
    # Weighted spread (standard deviation)
    if total_weight > 0:
        spread = np.sqrt(np.average((coords - centroid)**2, weights=weights, axis=0))
    else:
        spread = np.std(coords, axis=0)
    
    return {
        'positions': coords,
        'weights': weights,
        'centroid': centroid,
        'spread': spread,
    }


def topic_spatial_autocorrelation(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    topic_idx: int,
    radius: float = 50.0
) -> Dict[str, float]:
    """
    Compute Moran's I for topic weights to assess spatial clustering.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
    topic_idx : int
        Index of topic.
    radius : float, default 50.0
        Neighborhood radius for spatial weights.
        
    Returns
    -------
    dict
        Moran's I statistics including 'I', 'expected', 'zscore', 'pvalue'.
    """
    from spatialtissuepy.statistics.colocalization import morans_i
    
    topic_weights = model.transform(data)
    weights = topic_weights[:, topic_idx]
    
    return morans_i(data, weights, radius)


def topic_boundary_cells(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    topic_a: int,
    topic_b: int,
    threshold: float = 0.3,
    radius: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cells at the boundary between two topics.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
    topic_a, topic_b : int
        Topic indices.
    threshold : float, default 0.3
        Minimum weight for a cell to be considered part of a topic.
    radius : float, default 50.0
        Radius for finding neighbors.
        
    Returns
    -------
    boundary_a : np.ndarray
        Indices of topic_a cells near topic_b.
    boundary_b : np.ndarray
        Indices of topic_b cells near topic_a.
    """
    from scipy.spatial import cKDTree
    
    topic_weights = model.transform(data)
    coords = data._coordinates
    
    # Assign cells to topics based on dominant + threshold
    dominant = np.argmax(topic_weights, axis=1)
    max_weight = np.max(topic_weights, axis=1)
    
    cells_a = np.where((dominant == topic_a) & (max_weight >= threshold))[0]
    cells_b = np.where((dominant == topic_b) & (max_weight >= threshold))[0]
    
    if len(cells_a) == 0 or len(cells_b) == 0:
        return np.array([]), np.array([])
    
    # Find cells near the other topic
    tree_a = cKDTree(coords[cells_a])
    tree_b = cKDTree(coords[cells_b])
    
    # Cells in A near cells in B
    boundary_a_mask = np.zeros(len(cells_a), dtype=bool)
    for i, coord in enumerate(coords[cells_a]):
        neighbors = tree_b.query_ball_point(coord, radius)
        if len(neighbors) > 0:
            boundary_a_mask[i] = True
    
    # Cells in B near cells in A
    boundary_b_mask = np.zeros(len(cells_b), dtype=bool)
    for i, coord in enumerate(coords[cells_b]):
        neighbors = tree_a.query_ball_point(coord, radius)
        if len(neighbors) > 0:
            boundary_b_mask[i] = True
    
    boundary_a = cells_a[boundary_a_mask]
    boundary_b = cells_b[boundary_b_mask]
    
    return boundary_a, boundary_b


# -----------------------------------------------------------------------------
# Multi-Sample Analysis
# -----------------------------------------------------------------------------

def compare_topics_across_samples(
    model: 'SpatialLDA',
    samples: List['SpatialTissueData'],
    sample_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare topic prevalence across multiple samples.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    samples : list of SpatialTissueData
        Samples to compare.
    sample_ids : list of str, optional
        Names for samples.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with samples as rows and topic statistics as columns.
    """
    if sample_ids is None:
        sample_ids = [f'sample_{i}' for i in range(len(samples))]
    
    results = []
    
    for sample_id, sample in zip(sample_ids, samples):
        topic_weights = model.transform(sample)
        dominant = np.argmax(topic_weights, axis=1)
        
        row = {'sample_id': sample_id, 'n_cells': sample.n_cells}
        
        # Mean topic weights
        for i in range(model.n_topics):
            row[f'topic_{i}_mean'] = np.mean(topic_weights[:, i])
            row[f'topic_{i}_dominant_count'] = np.sum(dominant == i)
            row[f'topic_{i}_dominant_frac'] = np.mean(dominant == i)
        
        results.append(row)
    
    return pd.DataFrame(results)


def topic_prevalence_by_cell_type(
    model: 'SpatialLDA',
    data: 'SpatialTissueData'
) -> pd.DataFrame:
    """
    Analyze topic prevalence stratified by cell type.
    
    For each cell type, compute the distribution of topic assignments.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cell types as rows and topic statistics as columns.
    """
    topic_weights = model.transform(data)
    cell_types = data._cell_types
    unique_types = data.cell_types_unique
    
    results = []
    
    for cell_type in unique_types:
        mask = cell_types == cell_type
        type_weights = topic_weights[mask]
        
        row = {
            'cell_type': cell_type,
            'n_cells': np.sum(mask),
        }
        
        for i in range(model.n_topics):
            row[f'topic_{i}_mean'] = np.mean(type_weights[:, i])
            row[f'topic_{i}_std'] = np.std(type_weights[:, i])
        
        results.append(row)
    
    return pd.DataFrame(results)


def topic_transition_matrix(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    radius: float = 50.0
) -> pd.DataFrame:
    """
    Compute topic co-occurrence/transition matrix.
    
    For each pair of topics, count how often they appear in adjacent cells.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
    radius : float, default 50.0
        Neighborhood radius.
        
    Returns
    -------
    pd.DataFrame
        Topic co-occurrence matrix.
    """
    from scipy.spatial import cKDTree
    
    topic_weights = model.transform(data)
    dominant = np.argmax(topic_weights, axis=1)
    coords = data._coordinates
    
    # Build spatial neighbors
    tree = cKDTree(coords)
    
    # Count co-occurrences
    n_topics = model.n_topics
    co_occurrence = np.zeros((n_topics, n_topics))
    
    for i in range(data.n_cells):
        neighbors = tree.query_ball_point(coords[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        topic_i = dominant[i]
        
        for j in neighbors:
            topic_j = dominant[j]
            co_occurrence[topic_i, topic_j] += 1
    
    # Normalize by row sums
    row_sums = co_occurrence.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = co_occurrence / row_sums
        normalized = np.nan_to_num(normalized, nan=0)
    
    return pd.DataFrame(
        normalized,
        index=[f'Topic_{i}' for i in range(n_topics)],
        columns=[f'Topic_{i}' for i in range(n_topics)]
    )
