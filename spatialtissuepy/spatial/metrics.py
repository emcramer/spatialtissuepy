"""
Spatial operations metrics for integration with summary module.

This module registers spatial operation metrics with the StatisticsPanel
for standardized computation across samples.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from spatialtissuepy.summary.registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Distance Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mean_nearest_neighbor_distance',
    category='spatial',
    description='Mean distance to nearest neighbor (Clark-Evans statistic)',
    parameters={'k': int}
)
def _mean_nnd(data: 'SpatialTissueData', k: int = 1) -> Dict[str, float]:
    """Compute mean nearest neighbor distance."""
    from .distance import mean_nearest_neighbor_distance
    return {'mean_nearest_neighbor_distance': mean_nearest_neighbor_distance(data._coordinates, k=k)}


@register_metric(
    name='mean_nearest_neighbor_distance_by_type',
    category='spatial',
    description='Mean nearest neighbor distance within each cell type',
    parameters={'k': int}
)
def _mean_nnd_by_type(data: 'SpatialTissueData', k: int = 1) -> Dict[str, float]:
    """Compute mean NND for each cell type."""
    from .distance import mean_nearest_neighbor_distance
    
    result = {}
    for cell_type in data.cell_types_unique:
        idx = data.get_cells_by_type(cell_type)
        if len(idx) > k:
            coords = data._coordinates[idx]
            mnn = mean_nearest_neighbor_distance(coords, k=k)
            result[f'mean_nnd_{cell_type}'] = mnn
        else:
            result[f'mean_nnd_{cell_type}'] = np.nan
    return result


@register_metric(
    name='mean_distance_to_type',
    category='spatial',
    description='Mean distance from all cells to nearest cell of target type',
    parameters={'target_type': str}
)
def _mean_dist_to_type(
    data: 'SpatialTissueData', 
    target_type: str
) -> Dict[str, float]:
    """Compute mean distance to target cell type."""
    from .distance import distance_to_type
    
    try:
        distances = distance_to_type(data, target_type)
        return {f'mean_distance_to_{target_type}': float(np.mean(distances))}
    except ValueError:
        return {f'mean_distance_to_{target_type}': np.nan}


@register_metric(
    name='point_density',
    category='spatial',
    description='Cell density (cells per unit area)',
    parameters={'method': str}
)
def _point_density_metric(
    data: 'SpatialTissueData',
    method: str = 'bounding_box'
) -> Dict[str, float]:
    """Compute point density."""
    from .distance import point_density
    return {'point_density': point_density(data._coordinates, method=method)}


@register_metric(
    name='point_density_by_type',
    category='spatial',
    description='Cell density for each cell type',
    parameters={'method': str}
)
def _point_density_by_type(
    data: 'SpatialTissueData',
    method: str = 'bounding_box'
) -> Dict[str, float]:
    """Compute density for each cell type."""
    from .distance import point_density
    
    result = {}
    for cell_type in data.cell_types_unique:
        idx = data.get_cells_by_type(cell_type)
        if len(idx) >= 3:
            coords = data._coordinates[idx]
            density = point_density(coords, method=method)
            result[f'density_{cell_type}'] = density
        else:
            result[f'density_{cell_type}'] = np.nan
    return result


# -----------------------------------------------------------------------------
# Neighborhood Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mean_neighborhood_size',
    category='neighborhood',
    description='Mean number of neighbors per cell',
    parameters={'method': str, 'radius': float, 'k': int}
)
def _mean_neighborhood_size(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: float = 50.0,
    k: int = None
) -> Dict[str, float]:
    """Compute mean neighborhood size."""
    from .neighborhood import compute_neighborhoods, neighborhood_size
    
    neighborhoods = compute_neighborhoods(data, method=method, radius=radius, k=k)
    sizes = neighborhood_size(neighborhoods)
    return {
        'mean_neighborhood_size': float(np.mean(sizes)),
        'std_neighborhood_size': float(np.std(sizes)),
    }


@register_metric(
    name='mean_neighborhood_diversity',
    category='neighborhood',
    description='Mean Shannon diversity of cell type composition in neighborhoods',
    parameters={'method': str, 'radius': float, 'k': int}
)
def _mean_neighborhood_diversity(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: float = 50.0,
    k: int = None
) -> Dict[str, float]:
    """Compute mean neighborhood diversity."""
    from .neighborhood import compute_neighborhoods, neighborhood_diversity
    
    neighborhoods = compute_neighborhoods(data, method=method, radius=radius, k=k)
    diversity = neighborhood_diversity(data, neighborhoods, metric='shannon')
    return {'mean_neighborhood_diversity': float(np.mean(diversity))}


@register_metric(
    name='type_enrichment',
    category='neighborhood',
    description='Neighborhood enrichment of target type (observed/expected)',
    parameters={'target_type': str, 'method': str, 'radius': float}
)
def _type_enrichment(
    data: 'SpatialTissueData',
    target_type: str,
    method: str = 'radius',
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute mean neighborhood enrichment for target type."""
    from .neighborhood import compute_neighborhoods, neighborhood_enrichment
    
    neighborhoods = compute_neighborhoods(data, method=method, radius=radius)
    enrichment = neighborhood_enrichment(data, neighborhoods, target_type)
    return {
        f'mean_enrichment_{target_type}': float(np.mean(enrichment)),
        f'std_enrichment_{target_type}': float(np.std(enrichment)),
    }


@register_metric(
    name='interface_fraction',
    category='neighborhood',
    description='Fraction of cells at interface between two types',
    parameters={'type_a': str, 'type_b': str, 'radius': float, 'min_neighbors': int}
)
def _interface_fraction(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0,
    min_neighbors: int = 1
) -> Dict[str, float]:
    """Compute fraction of cells at type interface."""
    from .neighborhood import interface_cells
    
    try:
        a_interface, b_interface = interface_cells(
            data, type_a, type_b, radius, min_neighbors
        )
        
        n_a = len(data.get_cells_by_type(type_a))
        n_b = len(data.get_cells_by_type(type_b))
        
        return {
            f'interface_fraction_{type_a}': len(a_interface) / max(n_a, 1),
            f'interface_fraction_{type_b}': len(b_interface) / max(n_b, 1),
        }
    except Exception:
        return {
            f'interface_fraction_{type_a}': np.nan,
            f'interface_fraction_{type_b}': np.nan,
        }


# -----------------------------------------------------------------------------
# Clustering Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='n_spatial_clusters',
    category='spatial',
    description='Number of spatial clusters from DBSCAN',
    parameters={'eps': float, 'min_samples': int}
)
def _n_spatial_clusters(
    data: 'SpatialTissueData',
    eps: float = 30.0,
    min_samples: int = 5
) -> Dict[str, float]:
    """Count number of spatial clusters."""
    from .clustering import dbscan_clustering, cluster_statistics
    
    labels = dbscan_clustering(data, eps=eps, min_samples=min_samples)
    stats = cluster_statistics(data, labels)
    
    return {
        'n_spatial_clusters': stats['n_clusters'],
        'spatial_cluster_noise_fraction': stats['noise_fraction'],
    }


@register_metric(
    name='n_clusters_by_type',
    category='spatial',
    description='Number of spatial clusters for each cell type',
    parameters={'eps': float, 'min_samples': int}
)
def _n_clusters_by_type(
    data: 'SpatialTissueData',
    eps: float = 30.0,
    min_samples: int = 5
) -> Dict[str, float]:
    """Count clusters per cell type."""
    from .clustering import dbscan_by_type
    
    results = dbscan_by_type(data, eps=eps, min_samples=min_samples)
    
    output = {}
    for cell_type, labels in results.items():
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        output[f'n_clusters_{cell_type}'] = n_clusters
    
    return output


@register_metric(
    name='spatial_cluster_purity',
    category='spatial',
    description='Purity of spatial clusters by cell type',
    parameters={'eps': float, 'min_samples': int}
)
def _spatial_cluster_purity(
    data: 'SpatialTissueData',
    eps: float = 30.0,
    min_samples: int = 5
) -> Dict[str, float]:
    """Compute cluster purity."""
    from .clustering import dbscan_clustering, cluster_purity
    
    labels = dbscan_clustering(data, eps=eps, min_samples=min_samples)
    purity = cluster_purity(labels, data._cell_types)
    
    return {'spatial_cluster_purity': purity}


@register_metric(
    name='silhouette_score',
    category='spatial',
    description='Silhouette score for spatial clustering',
    parameters={'eps': float, 'min_samples': int, 'sample_size': int}
)
def _silhouette_score(
    data: 'SpatialTissueData',
    eps: float = 30.0,
    min_samples: int = 5,
    sample_size: int = 5000
) -> Dict[str, float]:
    """Compute silhouette score."""
    from .clustering import dbscan_clustering, silhouette_spatial
    
    labels = dbscan_clustering(data, eps=eps, min_samples=min_samples)
    
    # Check if we have enough clusters
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if len(unique_labels) < 2:
        return {'silhouette_score': np.nan}
    
    score = silhouette_spatial(data, labels, sample_size=sample_size)
    return {'silhouette_score': score}


@register_metric(
    name='n_connected_components_type',
    category='spatial',
    description='Number of spatially connected components for a cell type',
    parameters={'cell_type': str, 'radius': float}
)
def _n_connected_components_type(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float = 30.0
) -> Dict[str, float]:
    """Count connected components for a cell type."""
    from .clustering import connected_components_spatial
    
    try:
        labels = connected_components_spatial(
            data, radius=radius, cell_types=[cell_type]
        )
        n_components = len(set(labels[labels >= 0]))
        return {f'n_components_{cell_type}': n_components}
    except Exception:
        return {f'n_components_{cell_type}': np.nan}
