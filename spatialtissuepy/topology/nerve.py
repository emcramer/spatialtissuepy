"""
Nerve computation for Mapper algorithm.

The nerve of a cover is a simplicial complex (or graph) where:
- Each cluster within a cover element becomes a node
- Two nodes are connected if their clusters share points

This module handles clustering within cover elements and building
the resulting Mapper graph.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class MapperNode:
    """
    A node in the Mapper graph.
    
    Attributes
    ----------
    node_id : int
        Unique node identifier.
    members : np.ndarray
        Indices of cells belonging to this node.
    cover_element : int
        Index of the cover element this node came from.
    cluster_label : int
        Cluster label within the cover element.
    centroid : np.ndarray
        Centroid in feature space (neighborhood vectors).
    spatial_centroid : np.ndarray
        Centroid in spatial coordinates.
    """
    node_id: int
    members: np.ndarray
    cover_element: int
    cluster_label: int
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    spatial_centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def size(self) -> int:
        """Number of cells in this node."""
        return len(self.members)
    
    def __repr__(self) -> str:
        return f"MapperNode(id={self.node_id}, size={self.size}, cover={self.cover_element})"


@dataclass
class MapperEdge:
    """
    An edge in the Mapper graph.
    
    Attributes
    ----------
    source : int
        Source node ID.
    target : int
        Target node ID.
    weight : int
        Number of shared members (overlap size).
    shared_members : np.ndarray
        Indices of shared cells.
    """
    source: int
    target: int
    weight: int
    shared_members: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __repr__(self) -> str:
        return f"MapperEdge({self.source} -- {self.target}, weight={self.weight})"


def cluster_cover_element(
    element_members: np.ndarray,
    features: np.ndarray,
    method: str = 'dbscan',
    min_cluster_size: int = 3,
    **clustering_params
) -> np.ndarray:
    """
    Cluster points within a cover element.
    
    Parameters
    ----------
    element_members : np.ndarray
        Indices of points in this cover element.
    features : np.ndarray
        Feature matrix (n_points, n_features) for all points.
    method : str, default 'dbscan'
        Clustering method: 'dbscan', 'agglomerative', 'kmeans'.
    min_cluster_size : int, default 3
        Minimum points to form a cluster.
    **clustering_params
        Additional parameters for clustering algorithm.
    
    Returns
    -------
    np.ndarray
        Cluster labels for points in element_members.
        -1 indicates noise (no cluster).
    """
    if len(element_members) < min_cluster_size:
        return np.full(len(element_members), -1)
    
    element_features = features[element_members]
    
    if method == 'dbscan':
        from sklearn.cluster import DBSCAN
        
        eps = clustering_params.get('eps', 0.5)
        min_samples = clustering_params.get('min_samples', min_cluster_size)
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(element_features)
        
    elif method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        
        n_clusters = clustering_params.get('n_clusters', None)
        distance_threshold = clustering_params.get('distance_threshold', 0.5)
        linkage = clustering_params.get('linkage', 'ward')
        
        if n_clusters is None:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
        
        labels = clusterer.fit_predict(element_features)
        
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        
        n_clusters = clustering_params.get('n_clusters', 2)
        n_clusters = min(n_clusters, len(element_members))
        
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = clusterer.fit_predict(element_features)
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Filter small clusters
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        if np.sum(labels == label) < min_cluster_size:
            labels[labels == label] = -1
    
    return labels


def build_nerve(
    cover_element_members: List[np.ndarray],
    features: np.ndarray,
    coordinates: np.ndarray,
    clustering_method: str = 'dbscan',
    min_cluster_size: int = 3,
    min_edge_weight: int = 1,
    **clustering_params
) -> Tuple[List[MapperNode], List[MapperEdge]]:
    """
    Build the nerve (graph) from clustered cover elements.
    
    Parameters
    ----------
    cover_element_members : list of np.ndarray
        For each cover element, indices of points it contains.
    features : np.ndarray
        Feature matrix (n_points, n_features).
    coordinates : np.ndarray
        Spatial coordinates (n_points, n_dims).
    clustering_method : str, default 'dbscan'
        Clustering algorithm to use.
    min_cluster_size : int, default 3
        Minimum points per cluster.
    min_edge_weight : int, default 1
        Minimum overlap to create an edge.
    **clustering_params
        Parameters for clustering algorithm.
    
    Returns
    -------
    nodes : list of MapperNode
        Nodes in the Mapper graph.
    edges : list of MapperEdge
        Edges in the Mapper graph.
    """
    nodes: List[MapperNode] = []
    node_id_counter = 0
    
    # Track which cells belong to which nodes (for edge computation)
    cell_to_nodes: Dict[int, List[int]] = {}
    
    # Cluster each cover element
    for element_idx, members in enumerate(cover_element_members):
        if len(members) == 0:
            continue
        
        # Cluster within this element
        labels = cluster_cover_element(
            members,
            features,
            method=clustering_method,
            min_cluster_size=min_cluster_size,
            **clustering_params
        )
        
        # Create nodes for each cluster
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = labels == label
            cluster_members = members[cluster_mask]
            
            # Compute centroids
            centroid = features[cluster_members].mean(axis=0)
            spatial_centroid = coordinates[cluster_members].mean(axis=0)
            
            node = MapperNode(
                node_id=node_id_counter,
                members=cluster_members,
                cover_element=element_idx,
                cluster_label=label,
                centroid=centroid,
                spatial_centroid=spatial_centroid
            )
            nodes.append(node)
            
            # Track cell-to-node mapping
            for cell_idx in cluster_members:
                if cell_idx not in cell_to_nodes:
                    cell_to_nodes[cell_idx] = []
                cell_to_nodes[cell_idx].append(node_id_counter)
            
            node_id_counter += 1
    
    # Build edges from overlapping nodes
    edges: List[MapperEdge] = []
    edge_set: Set[Tuple[int, int]] = set()  # Track added edges
    
    # Count overlaps between all pairs of nodes
    overlap_counts: Dict[Tuple[int, int], List[int]] = {}
    
    for cell_idx, node_ids in cell_to_nodes.items():
        if len(node_ids) < 2:
            continue
        
        # All pairs of nodes sharing this cell
        for i, node_i in enumerate(node_ids):
            for node_j in node_ids[i + 1:]:
                edge_key = (min(node_i, node_j), max(node_i, node_j))
                if edge_key not in overlap_counts:
                    overlap_counts[edge_key] = []
                overlap_counts[edge_key].append(cell_idx)
    
    # Create edges for pairs with sufficient overlap
    for (node_i, node_j), shared_cells in overlap_counts.items():
        if len(shared_cells) >= min_edge_weight:
            edge = MapperEdge(
                source=node_i,
                target=node_j,
                weight=len(shared_cells),
                shared_members=np.array(shared_cells)
            )
            edges.append(edge)
    
    return nodes, edges


def nodes_edges_to_networkx(
    nodes: List[MapperNode],
    edges: List[MapperEdge],
    cell_types: Optional[np.ndarray] = None
) -> 'nx.Graph':
    """
    Convert Mapper nodes and edges to a NetworkX graph.
    
    Parameters
    ----------
    nodes : list of MapperNode
        Mapper nodes.
    edges : list of MapperEdge
        Mapper edges.
    cell_types : np.ndarray, optional
        Cell type labels for computing node compositions.
    
    Returns
    -------
    nx.Graph
        NetworkX graph with node and edge attributes.
    
    Raises
    ------
    ImportError
        If networkx is not installed.
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for graph operations. "
            "Install with: pip install networkx"
        )
    
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        node_attrs = {
            'size': node.size,
            'members': node.members,
            'cover_element': node.cover_element,
            'cluster_label': node.cluster_label,
            'centroid': node.centroid,
            'spatial_centroid': node.spatial_centroid,
        }
        
        # Add cell type composition if available
        if cell_types is not None:
            unique_types, counts = np.unique(
                cell_types[node.members], return_counts=True
            )
            composition = dict(zip(unique_types, counts))
            node_attrs['composition'] = composition
            
            # Dominant cell type
            dominant_idx = np.argmax(counts)
            node_attrs['dominant_type'] = unique_types[dominant_idx]
            node_attrs['dominant_fraction'] = counts[dominant_idx] / node.size
        
        G.add_node(node.node_id, **node_attrs)
    
    # Add edges with attributes
    for edge in edges:
        edge_attrs = {
            'weight': edge.weight,
            'shared_members': edge.shared_members,
        }
        G.add_edge(edge.source, edge.target, **edge_attrs)
    
    return G


def compute_graph_statistics(G: 'nx.Graph') -> Dict[str, Any]:
    """
    Compute summary statistics for a Mapper graph.
    
    Parameters
    ----------
    G : nx.Graph
        Mapper graph.
    
    Returns
    -------
    dict
        Dictionary of graph statistics.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for graph statistics")
    
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_connected_components': nx.number_connected_components(G),
    }
    
    if G.number_of_nodes() > 0:
        # Node degree statistics
        degrees = [d for n, d in G.degree()]
        stats['mean_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)
        
        # Node size statistics
        sizes = [G.nodes[n].get('size', 0) for n in G.nodes()]
        stats['mean_node_size'] = np.mean(sizes)
        stats['total_cells_in_nodes'] = np.sum(sizes)
        
        # Density
        if G.number_of_nodes() > 1:
            stats['density'] = nx.density(G)
        else:
            stats['density'] = 0.0
        
        # Clustering coefficient
        if G.number_of_nodes() > 2:
            stats['avg_clustering'] = nx.average_clustering(G)
        else:
            stats['avg_clustering'] = 0.0
    
    return stats
