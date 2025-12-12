"""
Spatial clustering methods for tissue analysis.

This module provides clustering algorithms that incorporate spatial location
information. Includes DBSCAN, HDBSCAN (optional), hierarchical clustering,
k-means with spatial features, and Leiden/Louvain community detection.

Key Algorithms
--------------
- DBSCAN: Density-based clustering, good for irregular shapes
- HDBSCAN: Hierarchical DBSCAN, handles varying density (optional dependency)
- Spatial k-means: k-means with spatial coordinates as features
- Hierarchical: Agglomerative clustering with distance constraints
- Leiden/Louvain: Graph-based community detection (uses network module)

References
----------
.. [1] Ester, M. et al. (1996). A density-based algorithm for discovering
       clusters in large spatial databases with noise. KDD.
.. [2] Campello, R. J. et al. (2013). Density-based clustering based on
       hierarchical density estimates. PAKDD.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Union, List, TYPE_CHECKING
from enum import Enum
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# Check for optional dependencies
try:
    import hdbscan as hdbscan_lib
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class ClusteringMethod(Enum):
    """Available clustering methods."""
    DBSCAN = 'dbscan'
    HDBSCAN = 'hdbscan'
    KMEANS = 'kmeans'
    HIERARCHICAL = 'hierarchical'
    SPATIAL_KMEANS = 'spatial_kmeans'
    LEIDEN = 'leiden'
    LOUVAIN = 'louvain'


# -----------------------------------------------------------------------------
# DBSCAN Clustering
# -----------------------------------------------------------------------------

def dbscan_clustering(
    data: 'SpatialTissueData',
    eps: float,
    min_samples: int = 5,
    cell_types: Optional[List[str]] = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    DBSCAN density-based spatial clustering.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    eps : float
        Maximum distance between samples in a neighborhood.
    min_samples : int, default 5
        Minimum samples in a neighborhood to form a core point.
    cell_types : list of str, optional
        Only cluster cells of these types. If None, cluster all.
    metric : str, default 'euclidean'
        Distance metric.

    Returns
    -------
    np.ndarray
        Cluster labels for each cell. -1 indicates noise.

    Notes
    -----
    DBSCAN is good for finding clusters of arbitrary shape and
    identifying outliers (noise points).

    Examples
    --------
    >>> labels = dbscan_clustering(data, eps=30, min_samples=5)
    >>> n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    """
    if cell_types is not None:
        mask = np.isin(data._cell_types, cell_types)
        coords = data._coordinates[mask]
    else:
        coords = data._coordinates
        mask = np.ones(data.n_cells, dtype=bool)
    
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    cluster_labels = clusterer.fit_predict(coords)
    
    # Map back to full array if subset was used
    if cell_types is not None:
        full_labels = np.full(data.n_cells, -1, dtype=int)
        full_labels[mask] = cluster_labels
        return full_labels
    
    return cluster_labels


def dbscan_by_type(
    data: 'SpatialTissueData',
    eps: float,
    min_samples: int = 5,
    metric: str = 'euclidean'
) -> Dict[str, np.ndarray]:
    """
    Run DBSCAN separately for each cell type.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    eps : float
        Maximum distance between samples.
    min_samples : int, default 5
        Minimum samples for core point.
    metric : str, default 'euclidean'
        Distance metric.

    Returns
    -------
    dict
        Dictionary mapping cell type to cluster labels.

    Notes
    -----
    Useful for finding spatial clusters within each cell population.
    """
    results = {}
    
    for cell_type in data.cell_types_unique:
        idx = data.get_cells_by_type(cell_type)
        coords = data._coordinates[idx]
        
        if len(coords) >= min_samples:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = clusterer.fit_predict(coords)
        else:
            labels = np.zeros(len(idx), dtype=int)
        
        results[cell_type] = labels
    
    return results


# -----------------------------------------------------------------------------
# HDBSCAN Clustering
# -----------------------------------------------------------------------------

def hdbscan_clustering(
    data: 'SpatialTissueData',
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cell_types: Optional[List[str]] = None,
    cluster_selection_method: str = 'eom'
) -> np.ndarray:
    """
    HDBSCAN hierarchical density-based clustering.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    min_cluster_size : int, default 10
        Minimum cluster size.
    min_samples : int, optional
        Minimum samples for core points. Default: min_cluster_size.
    cell_types : list of str, optional
        Only cluster cells of these types.
    cluster_selection_method : str, default 'eom'
        Method for selecting clusters: 'eom' or 'leaf'.

    Returns
    -------
    np.ndarray
        Cluster labels. -1 indicates noise.

    Notes
    -----
    HDBSCAN handles varying density better than DBSCAN and doesn't
    require specifying eps. Requires optional hdbscan package.

    Examples
    --------
    >>> labels = hdbscan_clustering(data, min_cluster_size=15)
    """
    if not HAS_HDBSCAN:
        raise ImportError(
            "hdbscan package required. Install with: pip install hdbscan"
        )
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    if cell_types is not None:
        mask = np.isin(data._cell_types, cell_types)
        coords = data._coordinates[mask]
    else:
        coords = data._coordinates
        mask = np.ones(data.n_cells, dtype=bool)
    
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
    )
    cluster_labels = clusterer.fit_predict(coords)
    
    if cell_types is not None:
        full_labels = np.full(data.n_cells, -1, dtype=int)
        full_labels[mask] = cluster_labels
        return full_labels
    
    return cluster_labels


# -----------------------------------------------------------------------------
# K-Means Clustering
# -----------------------------------------------------------------------------

def kmeans_spatial(
    data: 'SpatialTissueData',
    n_clusters: int,
    include_coords: bool = True,
    include_composition: bool = False,
    neighborhood_radius: Optional[float] = None,
    coord_weight: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    K-means clustering with spatial and/or neighborhood features.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_clusters : int
        Number of clusters.
    include_coords : bool, default True
        Include spatial coordinates in features.
    include_composition : bool, default False
        Include neighborhood composition in features.
    neighborhood_radius : float, optional
        Radius for computing neighborhood composition.
    coord_weight : float, default 1.0
        Weight for spatial coordinates relative to composition.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster labels.

    Examples
    --------
    >>> # Pure spatial clustering
    >>> labels = kmeans_spatial(data, n_clusters=10)
    >>> 
    >>> # Spatial + neighborhood composition
    >>> labels = kmeans_spatial(
    ...     data, n_clusters=10, 
    ...     include_composition=True, 
    ...     neighborhood_radius=50
    ... )
    """
    features = []
    
    if include_coords:
        # Scale coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(data._coordinates) * coord_weight
        features.append(coords_scaled)
    
    if include_composition:
        if neighborhood_radius is None:
            raise ValueError(
                "neighborhood_radius required when include_composition=True"
            )
        from spatialtissuepy.spatial.neighborhood import neighborhood_composition
        composition = neighborhood_composition(
            data, method='radius', radius=neighborhood_radius
        )
        features.append(composition)
    
    if not features:
        raise ValueError("At least one feature type must be enabled")
    
    X = np.hstack(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(X)


def kmeans_by_type(
    data: 'SpatialTissueData',
    n_clusters: int,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Run k-means separately for each cell type.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_clusters : int
        Number of clusters per type.
    random_state : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary mapping cell type to cluster labels.
    """
    results = {}
    
    for cell_type in data.cell_types_unique:
        idx = data.get_cells_by_type(cell_type)
        coords = data._coordinates[idx]
        
        if len(coords) >= n_clusters:
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(coords)
        else:
            labels = np.arange(len(idx))  # Each cell is own cluster
        
        results[cell_type] = labels
    
    return results


# -----------------------------------------------------------------------------
# Hierarchical Clustering
# -----------------------------------------------------------------------------

def hierarchical_clustering(
    data: 'SpatialTissueData',
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage_method: str = 'ward',
    cell_types: Optional[List[str]] = None
) -> np.ndarray:
    """
    Agglomerative hierarchical clustering.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_clusters : int, optional
        Number of clusters. Must specify either this or distance_threshold.
    distance_threshold : float, optional
        Distance threshold for cutting dendrogram.
    linkage_method : str, default 'ward'
        Linkage method: 'ward', 'complete', 'average', 'single'.
    cell_types : list of str, optional
        Only cluster cells of these types.

    Returns
    -------
    np.ndarray
        Cluster labels.

    Notes
    -----
    Ward linkage minimizes within-cluster variance and tends to create
    compact, spherical clusters.

    Examples
    --------
    >>> labels = hierarchical_clustering(data, n_clusters=20)
    >>> labels = hierarchical_clustering(data, distance_threshold=100)
    """
    if n_clusters is None and distance_threshold is None:
        raise ValueError("Specify either n_clusters or distance_threshold")
    
    if cell_types is not None:
        mask = np.isin(data._cell_types, cell_types)
        coords = data._coordinates[mask]
    else:
        coords = data._coordinates
        mask = np.ones(data.n_cells, dtype=bool)
    
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage_method
    )
    cluster_labels = clusterer.fit_predict(coords)
    
    if cell_types is not None:
        full_labels = np.full(data.n_cells, -1, dtype=int)
        full_labels[mask] = cluster_labels
        return full_labels
    
    return cluster_labels


def hierarchical_linkage(
    coordinates: np.ndarray,
    method: str = 'ward'
) -> np.ndarray:
    """
    Compute hierarchical clustering linkage matrix.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    method : str, default 'ward'
        Linkage method.

    Returns
    -------
    np.ndarray
        Linkage matrix for use with scipy.cluster.hierarchy functions.

    Notes
    -----
    Use scipy.cluster.hierarchy.dendrogram to visualize, and
    scipy.cluster.hierarchy.fcluster to extract clusters.
    """
    distances = pdist(coordinates)
    return linkage(distances, method=method)


def cut_dendrogram(
    Z: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Cut hierarchical clustering dendrogram to get cluster labels.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from hierarchical_linkage.
    n_clusters : int, optional
        Number of clusters.
    distance_threshold : float, optional
        Distance threshold.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    if n_clusters is not None:
        return fcluster(Z, t=n_clusters, criterion='maxclust')
    elif distance_threshold is not None:
        return fcluster(Z, t=distance_threshold, criterion='distance')
    else:
        raise ValueError("Specify n_clusters or distance_threshold")


# -----------------------------------------------------------------------------
# Graph-Based Clustering (Leiden/Louvain)
# -----------------------------------------------------------------------------

def leiden_clustering(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    resolution: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Leiden community detection on spatial graph.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Graph construction: 'radius' or 'knn'.
    radius : float, optional
        Neighborhood radius (for method='radius').
    k : int, optional
        Number of neighbors (for method='knn').
    resolution : float, default 1.0
        Resolution parameter. Higher = more smaller communities.
    random_state : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Community/cluster labels.

    Notes
    -----
    Requires leidenalg and igraph packages. Leiden improves on Louvain
    by guaranteeing well-connected communities.
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError(
            "leidenalg and igraph required. Install with: "
            "pip install leidenalg python-igraph"
        )
    
    # Build graph using network module
    from spatialtissuepy.network import CellGraph
    
    graph = CellGraph.from_spatial_data(
        data, method=method, radius=radius, k=k
    )
    
    # Convert to igraph
    edges = list(graph.graph.edges())
    g = ig.Graph(n=data.n_cells, edges=edges, directed=False)
    
    # Run Leiden
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state
    )
    
    return np.array(partition.membership)


def louvain_clustering(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    resolution: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Louvain community detection on spatial graph.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Graph construction method.
    radius : float, optional
        Neighborhood radius.
    k : int, optional
        Number of neighbors.
    resolution : float, default 1.0
        Resolution parameter.
    random_state : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Community labels.

    Notes
    -----
    Uses NetworkX's built-in Louvain implementation.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    
    from spatialtissuepy.network import CellGraph
    
    graph = CellGraph.from_spatial_data(
        data, method=method, radius=radius, k=k
    )
    
    # Run Louvain
    communities = louvain_communities(
        graph.graph, 
        resolution=resolution,
        seed=random_state
    )
    
    # Convert to label array
    labels = np.zeros(data.n_cells, dtype=int)
    for i, community in enumerate(communities):
        for node in community:
            labels[node] = i
    
    return labels


# -----------------------------------------------------------------------------
# Clustering Utilities
# -----------------------------------------------------------------------------

def cluster_statistics(
    data: 'SpatialTissueData',
    labels: np.ndarray
) -> Dict[str, Union[int, float, Dict]]:
    """
    Compute statistics for clustering results.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    dict
        Dictionary with clustering statistics.

    Examples
    --------
    >>> labels = dbscan_clustering(data, eps=30)
    >>> stats = cluster_statistics(data, labels)
    >>> print(stats['n_clusters'])
    15
    >>> print(stats['noise_fraction'])
    0.05
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Noise statistics
    noise_mask = labels == -1
    n_noise = np.sum(noise_mask)
    noise_fraction = n_noise / len(labels)
    
    # Cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[int(label)] = int(np.sum(labels == label))
    
    # Type composition per cluster
    type_composition = {}
    for label in unique_labels:
        if label != -1:
            mask = labels == label
            types, counts = np.unique(
                data._cell_types[mask], return_counts=True
            )
            type_composition[int(label)] = dict(zip(types, counts.astype(int)))
    
    # Spatial statistics per cluster
    cluster_centroids = {}
    cluster_radii = {}
    for label in unique_labels:
        if label != -1:
            mask = labels == label
            coords = data._coordinates[mask]
            centroid = np.mean(coords, axis=0)
            radius = np.max(np.linalg.norm(coords - centroid, axis=1))
            cluster_centroids[int(label)] = centroid.tolist()
            cluster_radii[int(label)] = float(radius)
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_fraction': noise_fraction,
        'cluster_sizes': cluster_sizes,
        'type_composition': type_composition,
        'cluster_centroids': cluster_centroids,
        'cluster_radii': cluster_radii,
        'mean_cluster_size': float(np.mean(list(cluster_sizes.values()))) if cluster_sizes else 0,
        'size_std': float(np.std(list(cluster_sizes.values()))) if cluster_sizes else 0,
    }


def cluster_purity(
    labels: np.ndarray,
    cell_types: np.ndarray
) -> float:
    """
    Compute cluster purity (how homogeneous clusters are by cell type).

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    cell_types : np.ndarray
        Cell type labels.

    Returns
    -------
    float
        Purity score in [0, 1]. Higher = more homogeneous clusters.

    Notes
    -----
    Purity = (1/N) * sum(max type count in each cluster)
    """
    unique_labels = np.unique(labels[labels != -1])
    
    if len(unique_labels) == 0:
        return 0.0
    
    total_correct = 0
    total_cells = 0
    
    for label in unique_labels:
        mask = labels == label
        _, counts = np.unique(cell_types[mask], return_counts=True)
        total_correct += np.max(counts)
        total_cells += np.sum(mask)
    
    return total_correct / total_cells


def silhouette_spatial(
    data: 'SpatialTissueData',
    labels: np.ndarray,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None
) -> float:
    """
    Compute silhouette score for spatial clustering.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    labels : np.ndarray
        Cluster labels.
    sample_size : int, optional
        Number of samples for approximation (for large datasets).
    random_state : int, optional
        Random seed for sampling.

    Returns
    -------
    float
        Mean silhouette score in [-1, 1]. Higher = better clustering.
    """
    from sklearn.metrics import silhouette_score
    
    # Remove noise points
    valid_mask = labels != -1
    coords = data._coordinates[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(np.unique(valid_labels)) < 2:
        return 0.0
    
    return silhouette_score(
        coords, 
        valid_labels,
        sample_size=sample_size,
        random_state=random_state
    )


# -----------------------------------------------------------------------------
# Spatial Regions
# -----------------------------------------------------------------------------

def spatial_regions(
    data: 'SpatialTissueData',
    n_regions: int,
    method: str = 'kmeans',
    **kwargs
) -> np.ndarray:
    """
    Divide tissue into spatial regions for analysis.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_regions : int
        Number of regions.
    method : str, default 'kmeans'
        Method: 'kmeans', 'hierarchical', or 'grid'.
    **kwargs
        Additional arguments passed to clustering method.

    Returns
    -------
    np.ndarray
        Region labels for each cell.

    Notes
    -----
    'grid' method creates rectangular regions based on coordinates.
    """
    if method == 'kmeans':
        return kmeans_spatial(data, n_clusters=n_regions, **kwargs)
    elif method == 'hierarchical':
        return hierarchical_clustering(data, n_clusters=n_regions, **kwargs)
    elif method == 'grid':
        return _grid_regions(data, n_regions)
    else:
        raise ValueError(f"Unknown method: {method}")


def _grid_regions(
    data: 'SpatialTissueData',
    n_regions: int
) -> np.ndarray:
    """Create grid-based regions."""
    bounds = data.bounds
    
    # Determine grid dimensions (roughly square)
    n_x = int(np.ceil(np.sqrt(n_regions)))
    n_y = int(np.ceil(n_regions / n_x))
    
    x_edges = np.linspace(bounds['x'][0], bounds['x'][1], n_x + 1)
    y_edges = np.linspace(bounds['y'][0], bounds['y'][1], n_y + 1)
    
    labels = np.zeros(data.n_cells, dtype=int)
    
    for i, (x, y) in enumerate(data._coordinates[:, :2]):
        x_bin = np.searchsorted(x_edges[1:], x, side='right')
        y_bin = np.searchsorted(y_edges[1:], y, side='right')
        x_bin = min(x_bin, n_x - 1)
        y_bin = min(y_bin, n_y - 1)
        labels[i] = y_bin * n_x + x_bin
    
    return labels


def connected_components_spatial(
    data: 'SpatialTissueData',
    radius: float,
    cell_types: Optional[List[str]] = None
) -> np.ndarray:
    """
    Find connected components based on spatial proximity.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Maximum distance for connectivity.
    cell_types : list of str, optional
        Only consider these cell types.

    Returns
    -------
    np.ndarray
        Component labels.
    """
    from spatialtissuepy.network import CellGraph
    import networkx as nx
    
    if cell_types is not None:
        subset = data.subset(cell_types=cell_types)
        graph = CellGraph.from_spatial_data(subset, method='radius', radius=radius)
        components = list(nx.connected_components(graph.graph))
        
        # Map back to original indices
        mask = np.isin(data._cell_types, cell_types)
        idx_map = np.where(mask)[0]
        
        labels = np.full(data.n_cells, -1, dtype=int)
        for i, component in enumerate(components):
            for node in component:
                labels[idx_map[node]] = i
        return labels
    
    graph = CellGraph.from_spatial_data(data, method='radius', radius=radius)
    components = list(nx.connected_components(graph.graph))
    
    labels = np.zeros(data.n_cells, dtype=int)
    for i, component in enumerate(components):
        for node in component:
            labels[node] = i
    
    return labels
