"""
Neighborhood and co-localization statistics for spatial summary.

These metrics describe local cell environments and cell-cell interactions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from .registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


@register_metric(
    name='mean_neighborhood_entropy',
    category='neighborhood',
    description='Mean Shannon entropy of neighborhood compositions',
    parameters={'radius': float}
)
def mean_neighborhood_entropy(
    data: 'SpatialTissueData',
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute mean neighborhood entropy across all cells.
    
    Higher entropy = more diverse neighborhoods.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    radius : float
        Neighborhood radius.
    
    Returns
    -------
    dict
        Keys: 'mean_neighborhood_entropy', 'std_neighborhood_entropy'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    unique_types = list(data.cell_types_unique)
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    if len(coords) < 2 or n_types < 2:
        return {
            'mean_neighborhood_entropy': np.nan,
            'std_neighborhood_entropy': np.nan
        }
    
    tree = cKDTree(coords)
    entropies = []
    
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            entropies.append(0.0)
            continue
        
        # Count types in neighborhood
        counts = np.zeros(n_types)
        for j in neighbors:
            counts[type_to_idx[cell_types[j]]] += 1
        
        # Entropy
        props = counts / counts.sum()
        props = props[props > 0]
        entropy = -np.sum(props * np.log(props))
        
        # Normalize by max possible
        max_entropy = np.log(n_types)
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        entropies.append(entropy)
    
    entropies = np.array(entropies)
    
    return {
        'mean_neighborhood_entropy': entropies.mean(),
        'std_neighborhood_entropy': entropies.std(),
    }


@register_metric(
    name='mean_neighborhood_composition',
    category='neighborhood',
    description='Mean proportion of each cell type in neighborhoods',
    parameters={'radius': float},
    dynamic_columns=True
)
def mean_neighborhood_composition(
    data: 'SpatialTissueData',
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute mean neighborhood composition.
    
    Returns
    -------
    dict
        Keys: 'mean_neighbor_prop_{type}' for each type.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    unique_types = list(data.cell_types_unique)
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    if len(coords) < 2:
        return {f'mean_neighbor_prop_{t}': np.nan for t in unique_types}
    
    tree = cKDTree(coords)
    
    # Accumulate proportions
    all_props = np.zeros((len(coords), n_types))
    
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            continue
        
        # Count types
        counts = np.zeros(n_types)
        for j in neighbors:
            counts[type_to_idx[cell_types[j]]] += 1
        
        all_props[i] = counts / counts.sum()
    
    mean_props = all_props.mean(axis=0)
    
    return {
        f'mean_neighbor_prop_{t}': mean_props[type_to_idx[t]]
        for t in unique_types
    }


@register_metric(
    name='neighborhood_homogeneity',
    category='neighborhood',
    description='Proportion of neighbors that are same type as focal cell',
    parameters={'radius': float}
)
def neighborhood_homogeneity(
    data: 'SpatialTissueData',
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute neighborhood homogeneity (same-type neighbor fraction).
    
    Higher = cells tend to be near same type.
    
    Returns
    -------
    dict
        Keys: 'mean_homogeneity', 'homogeneity_{type}'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    unique_types = list(data.cell_types_unique)
    
    if len(coords) < 2:
        result = {'mean_homogeneity': np.nan}
        result.update({f'homogeneity_{t}': np.nan for t in unique_types})
        return result
    
    tree = cKDTree(coords)
    
    same_type_fracs = []
    type_fracs = {t: [] for t in unique_types}
    
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            continue
        
        focal_type = cell_types[i]
        same_type = sum(1 for j in neighbors if cell_types[j] == focal_type)
        frac = same_type / len(neighbors)
        
        same_type_fracs.append(frac)
        type_fracs[focal_type].append(frac)
    
    result = {
        'mean_homogeneity': np.mean(same_type_fracs) if same_type_fracs else np.nan
    }
    
    for t in unique_types:
        if type_fracs[t]:
            result[f'homogeneity_{t}'] = np.mean(type_fracs[t])
        else:
            result[f'homogeneity_{t}'] = np.nan
    
    return result


@register_metric(
    name='colocalization_score',
    category='colocalization',
    description='Co-localization score between two cell types',
    parameters={'type_a': str, 'type_b': str, 'radius': float}
)
def colocalization_score(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute co-localization score between two cell types.
    
    Score = (observed A-B pairs) / (expected under CSR)
    
    > 1: co-localized (attract)
    = 1: random
    < 1: segregated (repel)
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    type_a, type_b : str
        Cell types to analyze.
    radius : float
        Interaction radius.
    
    Returns
    -------
    dict
        Key: 'coloc_{type_a}_{type_b}'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    n = len(coords)
    
    mask_a = cell_types == type_a
    mask_b = cell_types == type_b
    
    n_a = mask_a.sum()
    n_b = mask_b.sum()
    
    if n_a == 0 or n_b == 0:
        return {f'coloc_{type_a}_{type_b}': np.nan}
    
    coords_a = coords[mask_a]
    coords_b = coords[mask_b]
    
    # Count observed A-B pairs within radius
    tree_b = cKDTree(coords_b)
    observed = 0
    for coord_a in coords_a:
        n_neighbors = len(tree_b.query_ball_point(coord_a, radius))
        observed += n_neighbors
    
    # Expected under CSR
    extent = data.extent
    area = extent['x'] * extent['y']
    
    if area == 0:
        return {f'coloc_{type_a}_{type_b}': np.nan}
    
    # Expected pairs = n_a * n_b * (pi * r^2 / area)
    circle_area = np.pi * radius ** 2
    expected = n_a * n_b * (circle_area / area)
    
    score = observed / expected if expected > 0 else np.nan
    
    return {f'coloc_{type_a}_{type_b}': score}


@register_metric(
    name='mixing_score',
    category='colocalization',
    description='Mixing score between two cell types (normalized)',
    parameters={'type_a': str, 'type_b': str, 'radius': float}
)
def mixing_score(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute mixing score between two cell types.
    
    Mixing score = proportion of cross-type neighbors among all neighbors
    for cells of type A and B.
    
    1 = perfectly mixed
    0 = completely segregated
    
    Returns
    -------
    dict
        Key: 'mixing_{type_a}_{type_b}'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    
    mask_a = cell_types == type_a
    mask_b = cell_types == type_b
    mask_ab = mask_a | mask_b
    
    n_ab = mask_ab.sum()
    
    if n_ab < 2:
        return {f'mixing_{type_a}_{type_b}': np.nan}
    
    coords_ab = coords[mask_ab]
    types_ab = cell_types[mask_ab]
    
    tree = cKDTree(coords_ab)
    
    cross_type_neighbors = 0
    total_neighbors = 0
    
    for i, coord in enumerate(coords_ab):
        neighbors = tree.query_ball_point(coord, radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            continue
        
        focal_type = types_ab[i]
        for j in neighbors:
            total_neighbors += 1
            if types_ab[j] != focal_type:
                cross_type_neighbors += 1
    
    mixing = cross_type_neighbors / total_neighbors if total_neighbors > 0 else np.nan
    
    return {f'mixing_{type_a}_{type_b}': mixing}


@register_metric(
    name='interaction_strength_matrix',
    category='colocalization',
    description='Pairwise interaction strengths between all cell types',
    parameters={'radius': float},
    dynamic_columns=True
)
def interaction_strength_matrix(
    data: 'SpatialTissueData',
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute interaction strength for all cell type pairs.
    
    Returns
    -------
    dict
        Keys: 'interaction_{type_a}_{type_b}' for all pairs.
    """
    unique_types = list(data.cell_types_unique)
    result = {}
    
    for i, type_a in enumerate(unique_types):
        for type_b in unique_types[i:]:  # Upper triangle including diagonal
            coloc = colocalization_score(data, type_a, type_b, radius=radius)
            key_in = f'coloc_{type_a}_{type_b}'
            key_out = f'interaction_{type_a}_{type_b}'
            result[key_out] = coloc.get(key_in, np.nan)
    
    return result


@register_metric(
    name='border_contact_score',
    category='colocalization',
    description='Fraction of type A cells that contact type B',
    parameters={'type_a': str, 'type_b': str, 'contact_radius': float}
)
def border_contact_score(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    contact_radius: float = 20.0,
) -> Dict[str, float]:
    """
    Compute fraction of type A cells in contact with type B.
    
    Returns
    -------
    dict
        Key: 'contact_{type_a}_with_{type_b}'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    
    mask_a = cell_types == type_a
    mask_b = cell_types == type_b
    
    n_a = mask_a.sum()
    
    if n_a == 0 or mask_b.sum() == 0:
        return {f'contact_{type_a}_with_{type_b}': np.nan}
    
    coords_a = coords[mask_a]
    coords_b = coords[mask_b]
    
    tree_b = cKDTree(coords_b)
    
    # Count A cells with at least one B neighbor
    in_contact = 0
    for coord_a in coords_a:
        neighbors = tree_b.query_ball_point(coord_a, contact_radius)
        if len(neighbors) > 0:
            in_contact += 1
    
    fraction = in_contact / n_a
    
    return {f'contact_{type_a}_with_{type_b}': fraction}


@register_metric(
    name='infiltration_score',
    category='colocalization',
    description='Infiltration of type A into regions dominated by type B',
    parameters={'infiltrating_type': str, 'target_type': str, 'radius': float}
)
def infiltration_score(
    data: 'SpatialTissueData',
    infiltrating_type: str,
    target_type: str,
    radius: float = 100.0,
) -> Dict[str, float]:
    """
    Compute infiltration score.
    
    Measures how much infiltrating_type cells penetrate into
    regions dominated by target_type.
    
    Returns
    -------
    dict
        Key: 'infiltration_{infiltrating}_{target}'.
    """
    coords = data.coordinates
    cell_types = data.cell_types
    
    mask_inf = cell_types == infiltrating_type
    mask_target = cell_types == target_type
    
    if mask_inf.sum() == 0 or mask_target.sum() == 0:
        return {f'infiltration_{infiltrating_type}_into_{target_type}': np.nan}
    
    coords_inf = coords[mask_inf]
    coords_target = coords[mask_target]
    
    # For each infiltrating cell, compute local target density
    tree_target = cKDTree(coords_target)
    
    densities = []
    for coord in coords_inf:
        n_target_near = len(tree_target.query_ball_point(coord, radius))
        local_density = n_target_near / (np.pi * radius ** 2)
        densities.append(local_density)
    
    mean_density = np.mean(densities)
    
    # Normalize by global target density
    extent = data.extent
    global_density = mask_target.sum() / (extent['x'] * extent['y'])
    
    if global_density > 0:
        infiltration = mean_density / global_density
    else:
        infiltration = np.nan
    
    return {f'infiltration_{infiltrating_type}_into_{target_type}': infiltration}
