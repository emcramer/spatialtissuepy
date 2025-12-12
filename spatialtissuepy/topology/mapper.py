"""
Spatial Mapper algorithm implementation.

This module provides a lightweight, dependency-minimal implementation of
the Mapper algorithm from topological data analysis (TDA), specifically
designed for spatial tissue biology applications.

Key innovation: Spatial-aware filter functions that leverage cell coordinates
to capture tissue organization and gradients.

Example
-------
>>> from spatialtissuepy.topology import SpatialMapper
>>> from spatialtissuepy.topology.spatial_filters import distance_to_type_filter
>>>
>>> mapper = SpatialMapper(
...     filter_fn=distance_to_type_filter('Tumor'),
...     n_intervals=10,
...     overlap=0.5,
...     clustering='dbscan',
... )
>>> result = mapper.fit(data, neighborhood_radius=50)
>>> print(result)
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, 
    Tuple, Union
)
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

from .cover import Cover, UniformCover, AdaptiveCover, create_cover
from .nerve import (
    MapperNode, MapperEdge, build_nerve, 
    nodes_edges_to_networkx, compute_graph_statistics
)
from .filters import density_filter, pca_filter, FilterFunction

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData
    import networkx as nx


@dataclass
class MapperResult:
    """
    Container for Mapper algorithm results.
    
    Attributes
    ----------
    graph : nx.Graph
        NetworkX graph representation of the Mapper output.
    nodes : list of MapperNode
        List of Mapper nodes with metadata.
    edges : list of MapperEdge
        List of Mapper edges with overlap information.
    filter_values : np.ndarray
        Filter function values for each cell.
    cell_node_map : dict
        Mapping from cell index to list of node IDs.
    cover : Cover
        The cover used in the computation.
    parameters : dict
        Parameters used for this Mapper run.
    """
    graph: 'nx.Graph'
    nodes: List[MapperNode]
    edges: List[MapperEdge]
    filter_values: np.ndarray
    cell_node_map: Dict[int, List[int]]
    cover: Cover
    parameters: Dict[str, Any]
    
    # Cached statistics
    _statistics: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _node_compositions: Optional[Dict[int, Dict[str, int]]] = field(default=None, repr=False)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in the Mapper graph."""
        return len(self.nodes)
    
    @property
    def n_edges(self) -> int:
        """Number of edges in the Mapper graph."""
        return len(self.edges)
    
    @property
    def n_components(self) -> int:
        """Number of connected components."""
        try:
            import networkx as nx
            return nx.number_connected_components(self.graph)
        except ImportError:
            # Count manually from nodes/edges
            return self._count_components_manual()
    
    def _count_components_manual(self) -> int:
        """Count connected components without networkx."""
        if len(self.nodes) == 0:
            return 0
        
        # Union-find
        parent = {n.node_id: n.node_id for n in self.nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for edge in self.edges:
            union(edge.source, edge.target)
        
        return len(set(find(n.node_id) for n in self.nodes))
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Compute and cache graph statistics."""
        if self._statistics is None:
            try:
                self._statistics = compute_graph_statistics(self.graph)
            except ImportError:
                self._statistics = {
                    'n_nodes': self.n_nodes,
                    'n_edges': self.n_edges,
                    'n_connected_components': self.n_components,
                }
        return self._statistics
    
    @property
    def node_compositions(self) -> Dict[int, Dict[str, int]]:
        """Cell type composition of each node."""
        if self._node_compositions is None:
            self._node_compositions = {}
            for node in self.nodes:
                if hasattr(self.graph, 'nodes') and node.node_id in self.graph.nodes:
                    comp = self.graph.nodes[node.node_id].get('composition', {})
                    self._node_compositions[node.node_id] = comp
        return self._node_compositions
    
    @property
    def node_spatial_centroids(self) -> Dict[int, np.ndarray]:
        """Spatial centroids of each node."""
        return {node.node_id: node.spatial_centroid for node in self.nodes}
    
    def get_node_members(self, node_id: int) -> np.ndarray:
        """
        Get cell indices belonging to a specific node.
        
        Parameters
        ----------
        node_id : int
            Node identifier.
        
        Returns
        -------
        np.ndarray
            Array of cell indices.
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return node.members.copy()
        raise ValueError(f"Node {node_id} not found")
    
    def get_cells_by_component(self, component_id: int = 0) -> np.ndarray:
        """
        Get all cells in a connected component.
        
        Parameters
        ----------
        component_id : int, default 0
            Component index (0 is largest component).
        
        Returns
        -------
        np.ndarray
            Array of cell indices in the component.
        """
        try:
            import networkx as nx
            components = list(nx.connected_components(self.graph))
            # Sort by size (descending)
            components = sorted(components, key=len, reverse=True)
            
            if component_id >= len(components):
                raise ValueError(f"Component {component_id} not found (only {len(components)} components)")
            
            component_nodes = components[component_id]
            cells = []
            for node_id in component_nodes:
                cells.extend(self.get_node_members(node_id))
            
            return np.unique(cells)
        except ImportError:
            raise ImportError("networkx required for get_cells_by_component")
    
    def __repr__(self) -> str:
        return (
            f"MapperResult(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"n_components={self.n_components})"
        )
    
    def __str__(self) -> str:
        lines = [
            "MapperResult",
            f"  Nodes: {self.n_nodes}",
            f"  Edges: {self.n_edges}",
            f"  Connected components: {self.n_components}",
        ]
        
        stats = self.statistics
        if 'mean_degree' in stats:
            lines.append(f"  Mean node degree: {stats['mean_degree']:.2f}")
        if 'mean_node_size' in stats:
            lines.append(f"  Mean node size: {stats['mean_node_size']:.1f} cells")
        if 'total_cells_in_nodes' in stats:
            lines.append(f"  Total cells in graph: {stats['total_cells_in_nodes']}")
        
        return "\n".join(lines)


class SpatialMapper:
    """
    Spatial Mapper algorithm for cell community discovery.
    
    Implements the Mapper algorithm from topological data analysis with
    spatial-aware filter functions designed for tissue biology.
    
    Parameters
    ----------
    filter_fn : str, callable, or FilterFunction
        Filter function to use:
        - 'density': Local cell density
        - 'pca': First principal component
        - callable: Custom filter function
    cover_type : str, default 'uniform'
        Type of cover: 'uniform' or 'adaptive'.
    n_intervals : int, default 10
        Number of intervals in the cover.
    overlap : float, default 0.5
        Overlap fraction between intervals.
    clustering : str, default 'dbscan'
        Clustering algorithm: 'dbscan', 'agglomerative', 'kmeans'.
    clustering_params : dict, optional
        Parameters for clustering algorithm.
    min_cluster_size : int, default 3
        Minimum cells to form a cluster.
    min_edge_weight : int, default 1
        Minimum overlap to create an edge.
    
    Examples
    --------
    >>> mapper = SpatialMapper(
    ...     filter_fn='density',
    ...     n_intervals=10,
    ...     overlap=0.5,
    ... )
    >>> result = mapper.fit(data, neighborhood_radius=50)
    
    >>> # With spatial filter
    >>> from spatialtissuepy.topology.spatial_filters import radial_filter
    >>> mapper = SpatialMapper(
    ...     filter_fn=radial_filter(center=[500, 500]),
    ...     n_intervals=12,
    ...     overlap=0.4,
    ... )
    >>> result = mapper.fit(data, neighborhood_radius=50)
    """
    
    def __init__(
        self,
        filter_fn: Union[str, FilterFunction] = 'density',
        cover_type: str = 'uniform',
        n_intervals: int = 10,
        overlap: float = 0.5,
        clustering: str = 'dbscan',
        clustering_params: Optional[Dict[str, Any]] = None,
        min_cluster_size: int = 3,
        min_edge_weight: int = 1,
    ):
        self.filter_fn = filter_fn
        self.cover_type = cover_type
        self.n_intervals = n_intervals
        self.overlap = overlap
        self.clustering = clustering
        self.clustering_params = clustering_params or {}
        self.min_cluster_size = min_cluster_size
        self.min_edge_weight = min_edge_weight
        
        # Resolve string filter to function
        self._filter_fn = self._resolve_filter(filter_fn)
    
    def _resolve_filter(
        self,
        filter_fn: Union[str, FilterFunction]
    ) -> FilterFunction:
        """Resolve filter string to function."""
        if callable(filter_fn):
            return filter_fn
        
        filter_map = {
            'density': density_filter(),
            'pca': pca_filter(n_components=1),
        }
        
        if filter_fn in filter_map:
            return filter_map[filter_fn]
        
        raise ValueError(
            f"Unknown filter: {filter_fn}. "
            f"Options: {list(filter_map.keys())} or provide a callable."
        )
    
    def fit(
        self,
        data: 'SpatialTissueData',
        neighborhood_radius: float = 50.0,
        features: Optional[np.ndarray] = None,
    ) -> MapperResult:
        """
        Fit Mapper to spatial tissue data.
        
        Parameters
        ----------
        data : SpatialTissueData
            Input spatial data.
        neighborhood_radius : float, default 50.0
            Radius for computing neighborhood compositions.
        features : np.ndarray, optional
            Precomputed feature matrix. If None, computes neighborhood
            composition matrix.
        
        Returns
        -------
        MapperResult
            Mapper results including graph, nodes, edges.
        """
        coordinates = data._coordinates
        cell_types = data._cell_types
        
        # Compute neighborhood composition matrix if not provided
        if features is None:
            features = self._compute_neighborhood_matrix(
                data, radius=neighborhood_radius
            )
        
        # Compute filter values
        filter_values = self._filter_fn(coordinates, features, data)
        
        # Create and fit cover
        cover = create_cover(
            cover_type=self.cover_type,
            n_intervals=self.n_intervals,
            overlap_fraction=self.overlap
        )
        cover.fit(filter_values)
        
        # Get cover element members
        cover_members = cover.get_element_members(filter_values)
        
        # Build nerve (cluster and connect)
        nodes, edges = build_nerve(
            cover_element_members=cover_members,
            features=features,
            coordinates=coordinates,
            clustering_method=self.clustering,
            min_cluster_size=self.min_cluster_size,
            min_edge_weight=self.min_edge_weight,
            **self.clustering_params
        )
        
        # Convert to NetworkX graph
        try:
            graph = nodes_edges_to_networkx(nodes, edges, cell_types)
        except ImportError:
            # Create minimal graph placeholder
            graph = None
        
        # Build cell-to-node mapping
        cell_node_map: Dict[int, List[int]] = {}
        for node in nodes:
            for cell_idx in node.members:
                if cell_idx not in cell_node_map:
                    cell_node_map[cell_idx] = []
                cell_node_map[cell_idx].append(node.node_id)
        
        # Store parameters
        parameters = {
            'filter_fn': str(self.filter_fn),
            'cover_type': self.cover_type,
            'n_intervals': self.n_intervals,
            'overlap': self.overlap,
            'clustering': self.clustering,
            'clustering_params': self.clustering_params,
            'min_cluster_size': self.min_cluster_size,
            'min_edge_weight': self.min_edge_weight,
            'neighborhood_radius': neighborhood_radius,
        }
        
        return MapperResult(
            graph=graph,
            nodes=nodes,
            edges=edges,
            filter_values=filter_values,
            cell_node_map=cell_node_map,
            cover=cover,
            parameters=parameters,
        )
    
    def _compute_neighborhood_matrix(
        self,
        data: 'SpatialTissueData',
        radius: float
    ) -> np.ndarray:
        """
        Compute neighborhood composition matrix.
        
        For each cell, counts the number of each cell type within radius.
        
        Parameters
        ----------
        data : SpatialTissueData
            Input data.
        radius : float
            Neighborhood radius.
        
        Returns
        -------
        np.ndarray
            Shape (n_cells, n_cell_types) neighborhood composition matrix.
        """
        coordinates = data._coordinates
        cell_types = data._cell_types
        unique_types = data.cell_types_unique
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        
        n_cells = len(coordinates)
        n_types = len(unique_types)
        
        # Build KD-tree
        tree = cKDTree(coordinates)
        
        # Compute neighborhoods
        neighborhoods = np.zeros((n_cells, n_types), dtype=float)
        
        for i, coord in enumerate(coordinates):
            # Find neighbors within radius
            neighbor_idx = tree.query_ball_point(coord, radius)
            
            # Count cell types (excluding self)
            for j in neighbor_idx:
                if j != i:
                    type_idx = type_to_idx[cell_types[j]]
                    neighborhoods[i, type_idx] += 1
        
        # Normalize rows (optional: convert to proportions)
        row_sums = neighborhoods.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        neighborhoods = neighborhoods / row_sums
        
        return neighborhoods
    
    def __repr__(self) -> str:
        return (
            f"SpatialMapper(filter_fn={self.filter_fn!r}, "
            f"n_intervals={self.n_intervals}, overlap={self.overlap})"
        )


def spatial_mapper(
    data: 'SpatialTissueData',
    filter_fn: Union[str, FilterFunction] = 'density',
    neighborhood_radius: float = 50.0,
    n_intervals: int = 10,
    overlap: float = 0.5,
    clustering: str = 'dbscan',
    clustering_params: Optional[Dict[str, Any]] = None,
    min_cluster_size: int = 3,
) -> MapperResult:
    """
    Convenience function to run Spatial Mapper.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input spatial data.
    filter_fn : str or callable, default 'density'
        Filter function.
    neighborhood_radius : float, default 50.0
        Radius for neighborhood computation.
    n_intervals : int, default 10
        Number of cover intervals.
    overlap : float, default 0.5
        Cover overlap fraction.
    clustering : str, default 'dbscan'
        Clustering algorithm.
    clustering_params : dict, optional
        Clustering parameters.
    min_cluster_size : int, default 3
        Minimum cluster size.
    
    Returns
    -------
    MapperResult
        Mapper results.
    
    Examples
    --------
    >>> result = spatial_mapper(data, filter_fn='density', n_intervals=10)
    >>> print(f"Found {result.n_nodes} community nodes")
    """
    mapper = SpatialMapper(
        filter_fn=filter_fn,
        n_intervals=n_intervals,
        overlap=overlap,
        clustering=clustering,
        clustering_params=clustering_params,
        min_cluster_size=min_cluster_size,
    )
    return mapper.fit(data, neighborhood_radius=neighborhood_radius)
