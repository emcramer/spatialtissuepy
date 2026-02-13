"""
Evaluation metrics for Spatial LDA models.

This module provides metrics for assessing the quality and interpretability
of fitted Spatial LDA models.
"""

from __future__ import annotations
from typing import Optional, List, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    from .spatial_lda import SpatialLDA


# -----------------------------------------------------------------------------
# Topic Quality Metrics
# -----------------------------------------------------------------------------

def topic_coherence(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    n_top_types: int = 5,
    method: str = 'pmi',
    return_aggregate: bool = True
) -> Union[float, Dict[int, float]]:
    """
    Compute topic coherence scores.
    
    Measures how semantically coherent the top cell types in each topic are,
    based on their co-occurrence in neighborhoods.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to compute co-occurrence from.
    n_top_types : int, default 5
        Number of top cell types per topic to consider.
    method : str, default 'pmi'
        Coherence method: 'pmi' (pointwise mutual information) or 'npmi' (normalized).
    return_aggregate : bool, default True
        If True, return mean coherence across all topics.
        If False, return dict of per-topic scores.
        
    Returns
    -------
    float or dict
        Mean coherence score or mapping from topic index to score.
        
    Notes
    -----
    Higher coherence indicates more interpretable topics where the
    top cell types frequently co-occur in neighborhoods.
    """
    from spatialtissuepy.spatial.neighborhood import compute_neighborhoods, neighborhood_counts

    # Get neighborhoods present
    neighborhoods = compute_neighborhoods(
        data,
        method=model.neighborhood_method,
        radius=model.neighborhood_radius,
        k=model.neighborhood_k,
        include_self=True
    )
    
    # Get neighborhood counts
    counts = neighborhood_counts(
        data,
        neighborhoods
    )
    
    n_cells = counts.shape[0]
    cell_types = list(data.cell_types_unique)
    n_types = len(cell_types)
    
    # Compute co-occurrence matrix
    cooccur = np.zeros((n_types, n_types))
    marginal = np.zeros(n_types)
    
    for i in range(n_cells):
        present = counts[i] > 0
        present_idx = np.where(present)[0]
        
        for idx in present_idx:
            marginal[idx] += 1
            for idx2 in present_idx:
                cooccur[idx, idx2] += 1
    
    # Normalize
    p_joint = cooccur / n_cells
    p_marginal = marginal / n_cells
    
    # Compute coherence for each topic
    coherence_scores = {}
    
    for topic_idx in range(model.n_topics):
        # Get top cell types for this topic
        topic_weights = model.topic_cell_type_matrix_[topic_idx]
        
        # Map to data's cell type indices
        type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
        model_weights = np.zeros(n_types)
        
        for i, ct in enumerate(model.cell_types_):
            if ct in type_to_idx:
                model_weights[type_to_idx[ct]] = topic_weights[i]
        
        top_indices = np.argsort(model_weights)[::-1][:n_top_types]
        
        # Compute pairwise coherence
        coherence_sum = 0.0
        n_pairs = 0
        
        for i, idx1 in enumerate(top_indices):
            for idx2 in top_indices[i+1:]:
                p_xy = p_joint[idx1, idx2]
                p_x = p_marginal[idx1]
                p_y = p_marginal[idx2]
                
                if p_x > 0 and p_y > 0 and p_xy > 0:
                    if method == 'pmi':
                        # Pointwise mutual information
                        coherence_sum += np.log(p_xy / (p_x * p_y))
                    elif method == 'npmi':
                        # Normalized PMI
                        pmi = np.log(p_xy / (p_x * p_y))
                        npmi = pmi / (-np.log(p_xy))
                        coherence_sum += npmi
                
                n_pairs += 1
        
        if n_pairs > 0:
            coherence_scores[topic_idx] = coherence_sum / n_pairs
        else:
            coherence_scores[topic_idx] = 0.0
    
    if return_aggregate:
        return float(np.mean(list(coherence_scores.values())))
    
    return coherence_scores


def topic_diversity(
    model: 'SpatialLDA',
    n_top_types: int = 10
) -> float:
    """
    Compute topic diversity score.
    
    Measures how distinct topics are from each other by looking at
    overlap in their top cell types.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    n_top_types : int, default 10
        Number of top cell types per topic to consider.
        
    Returns
    -------
    float
        Diversity score between 0 (all topics identical) and 1 (no overlap).
        
    Notes
    -----
    Higher diversity indicates more distinct, interpretable topics.
    """
    if not model._is_fitted:
        raise RuntimeError("Model not fitted.")
    
    # Get top types for each topic
    top_types_per_topic = []
    
    for topic_idx in range(model.n_topics):
        weights = model.topic_cell_type_matrix_[topic_idx]
        top_indices = np.argsort(weights)[::-1][:n_top_types]
        top_types = set(top_indices)
        top_types_per_topic.append(top_types)
    
    # Compute pairwise Jaccard distances
    distances = []
    
    for i in range(model.n_topics):
        for j in range(i + 1, model.n_topics):
            intersection = len(top_types_per_topic[i] & top_types_per_topic[j])
            union = len(top_types_per_topic[i] | top_types_per_topic[j])
            
            if union > 0:
                jaccard = intersection / union
                distances.append(1 - jaccard)  # Distance = 1 - similarity
    
    if len(distances) == 0:
        return 1.0
    
    return np.mean(distances)


def topic_exclusivity(
    model: 'SpatialLDA',
    n_top_types: int = 10
) -> Dict[int, float]:
    """
    Compute exclusivity score for each topic.
    
    Measures how exclusive the top cell types are to each topic.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    n_top_types : int, default 10
        Number of top cell types to consider.
        
    Returns
    -------
    dict
        Mapping from topic index to exclusivity score.
    """
    if not model._is_fitted:
        raise RuntimeError("Model not fitted.")
    
    topic_matrix = model.topic_cell_type_matrix_
    n_topics, n_types = topic_matrix.shape
    
    exclusivity_scores = {}
    
    for topic_idx in range(n_topics):
        weights = topic_matrix[topic_idx]
        top_indices = np.argsort(weights)[::-1][:n_top_types]
        
        # Compute exclusivity for top types
        excl_sum = 0.0
        
        for type_idx in top_indices:
            # Weight in this topic vs sum across all topics
            weight_in_topic = topic_matrix[topic_idx, type_idx]
            total_weight = np.sum(topic_matrix[:, type_idx])
            
            if total_weight > 0:
                excl_sum += weight_in_topic / total_weight
        
        exclusivity_scores[topic_idx] = excl_sum / n_top_types
    
    return exclusivity_scores


# -----------------------------------------------------------------------------
# Spatial Consistency Metrics
# -----------------------------------------------------------------------------

def spatial_topic_consistency(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    radius: float = 50.0
) -> Dict[str, float]:
    """
    Measure spatial consistency of topic assignments.
    
    Checks if spatially nearby cells have similar topic assignments.
    
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
    dict
        Dictionary with:
        - 'agreement_rate': Fraction of neighbor pairs with same dominant topic
        - 'topic_autocorrelation': Average Moran's I across topics
        - 'spatial_entropy_reduction': How much entropy is reduced spatially
    """
    topic_weights = model.transform(data)
    dominant = np.argmax(topic_weights, axis=1)
    coords = data._coordinates
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Compute agreement rate
    agreements = 0
    total_pairs = 0
    
    for i in range(data.n_cells):
        neighbors = tree.query_ball_point(coords[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        for j in neighbors:
            if dominant[i] == dominant[j]:
                agreements += 1
            total_pairs += 1
    
    agreement_rate = agreements / total_pairs if total_pairs > 0 else 0
    
    # Compute autocorrelation for each topic
    morans_i_values = []
    
    for topic_idx in range(model.n_topics):
        from spatialtissuepy.statistics.colocalization import morans_i
        result = morans_i(data, topic_weights[:, topic_idx], radius)
        morans_i_values.append(result['I'])
    
    topic_autocorrelation = np.mean(morans_i_values)
    
    # Compute entropy
    def entropy(weights):
        with np.errstate(divide='ignore', invalid='ignore'):
            log_w = np.log2(weights + 1e-10)
            return -np.sum(weights * log_w, axis=1)
    
    cell_entropy = entropy(topic_weights)
    
    # Compute neighborhood-averaged entropy
    neighbor_entropy = np.zeros(data.n_cells)
    for i in range(data.n_cells):
        neighbors = tree.query_ball_point(coords[i], radius)
        if len(neighbors) > 0:
            avg_weights = np.mean(topic_weights[neighbors], axis=0)
            neighbor_entropy[i] = -np.sum(avg_weights * np.log2(avg_weights + 1e-10))
    
    entropy_reduction = np.mean(cell_entropy - neighbor_entropy)
    
    return {
        'agreement_rate': agreement_rate,
        'topic_autocorrelation': topic_autocorrelation,
        'spatial_entropy_reduction': entropy_reduction,
    }


def topic_concentration_index(
    model: 'SpatialLDA',
    data: 'SpatialTissueData'
) -> Dict[int, float]:
    """
    Compute spatial concentration index for each topic.
    
    Measures how spatially concentrated each topic is (vs uniformly spread).
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted model.
    data : SpatialTissueData
        Data to analyze.
        
    Returns
    -------
    dict
        Concentration index for each topic (higher = more concentrated).
    """
    topic_weights = model.transform(data)
    coords = data._coordinates
    
    concentration = {}
    
    for topic_idx in range(model.n_topics):
        weights = topic_weights[:, topic_idx]
        
        if np.sum(weights) < 1e-10:
            concentration[topic_idx] = 0.0
            continue
        
        # Weighted centroid
        centroid = np.average(coords, weights=weights, axis=0)
        
        # Weighted distance from centroid
        distances = np.linalg.norm(coords - centroid, axis=1)
        weighted_dist = np.average(distances, weights=weights)
        
        # Compare to uniform spread
        uniform_dist = np.mean(distances)
        
        # Concentration = 1 - (weighted_dist / uniform_dist)
        if uniform_dist > 0:
            concentration[topic_idx] = max(0, 1 - weighted_dist / uniform_dist)
        else:
            concentration[topic_idx] = 0.0
    
    return concentration


# -----------------------------------------------------------------------------
# Model Selection Metrics
# -----------------------------------------------------------------------------

def compute_model_selection_metrics(
    model: Union['SpatialLDA', List[int]],
    data: Optional['SpatialTissueData'] = None,
    n_topics_range: Optional[List[int]] = None,
    neighborhood_radius: float = 50.0,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute metrics for model selection (choosing number of topics).
    
    Parameters
    ----------
    model : SpatialLDA or list of int
        If SpatialLDA, it's used as a template.
        If list, it's treated as n_topics_range.
    data : SpatialTissueData
        Data to fit.
    n_topics_range : list of int, optional
        Range of topic numbers to try.
    neighborhood_radius : float, default 50.0
        Radius for neighborhoods.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each n_topics value.
    """
    from .spatial_lda import SpatialLDA
    
    # Handle argument polymorphic signature from tests
    if isinstance(model, list):
        n_topics_range = model
        # data should be the second arg
    elif data is None and n_topics_range is not None:
        # data was passed as n_topics_range? No, that's unlikely.
        pass

    if n_topics_range is None:
        n_topics_range = [3, 5, 7, 10]
    
    results = []
    
    for n_topics in n_topics_range:
        m = SpatialLDA(
            n_topics=n_topics,
            neighborhood_radius=neighborhood_radius,
            random_state=random_state,
        )
        
        m.fit(data)
        
        # Compute metrics
        perplexity = m.perplexity(data)
        log_likelihood = m.score(data)
        diversity = topic_diversity(m)
        mean_coherence = topic_coherence(m, data, return_aggregate=True)
        
        results.append({
            'n_topics': n_topics,
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'diversity': diversity,
            'mean_coherence': mean_coherence,
        })
    
    return pd.DataFrame(results)
