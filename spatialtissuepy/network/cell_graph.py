"""
CellGraph class for spatial tissue network analysis.

Provides a high-level interface for building and analyzing cell graphs
from spatial tissue data.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
)
import numpy as np

from .graph_construction import (
    GraphMethod, build_graph, _check_networkx, HAS_NETWORKX
)

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData
    import networkx as nx


class CellGraph:
    """
    A graph representation of spatial tissue data.
    
    CellGraph wraps a NetworkX graph with convenience methods for
    analyzing cell-cell relationships in tissue samples.
    
    Parameters
    ----------
    graph : nx.Graph
        The underlying NetworkX graph.
    cell_types : np.ndarray
        Cell type labels for each node.
    coordinates : np.ndarray
        Spatial coordinates for each node.
    method : str
        Graph construction method used.
    params : dict
        Parameters used for graph construction.
    
    Attributes
    ----------
    G : nx.Graph
        The underlying NetworkX graph.
    n_nodes : int
        Number of nodes (cells).
    n_edges : int
        Number of edges.
    cell_types_unique : list
        Unique cell types in the graph.
    
    Examples
    --------
    >>> from spatialtissuepy.network import CellGraph
    >>> 
    >>> # From SpatialTissueData
    >>> graph = CellGraph.from_spatial_data(
    ...     data,
    ...     method='proximity',
    ...     radius=30.0
    ... )
    >>> 
    >>> # Analyze
    >>> print(f"Nodes: {graph.n_nodes}, Edges: {graph.n_edges}")
    >>> centrality = graph.degree_centrality()
    """
    
    def __init__(
        self,
        graph: 'nx.Graph',
        cell_types: np.ndarray,
        coordinates: np.ndarray,
        method: str = 'unknown',
        params: Optional[Dict[str, Any]] = None,
    ):
        _check_networkx()
        
        self._G = graph
        self._cell_types = np.asarray(cell_types)
        self._coordinates = np.asarray(coordinates)
        self._method = method
        self._params = params or {}
        
        # Cache for computed metrics
        self._cache: Dict[str, Any] = {}
    
    @classmethod
    def from_spatial_data(
        cls,
        data: 'SpatialTissueData',
        method: Union[str, GraphMethod] = 'proximity',
        radius: float = 50.0,
        k: int = 6,
        mutual_knn: bool = False,
        max_edge_length: Optional[float] = None,
    ) -> 'CellGraph':
        """
        Create a CellGraph from SpatialTissueData.
        
        Parameters
        ----------
        data : SpatialTissueData
            Input spatial data.
        method : str or GraphMethod, default 'proximity'
            Graph construction method:
            - 'proximity': Connect cells within radius
            - 'knn': k-nearest neighbors
            - 'delaunay': Delaunay triangulation
            - 'gabriel': Gabriel graph
        radius : float, default 50.0
            Radius for proximity graph.
        k : int, default 6
            Number of neighbors for kNN graph.
        mutual_knn : bool, default False
            Use mutual kNN.
        max_edge_length : float, optional
            Maximum edge length for Delaunay pruning.
        
        Returns
        -------
        CellGraph
            Cell graph built from the spatial data.
        """
        coordinates = data.coordinates
        cell_types = data.cell_types
        
        # Build graph
        G = build_graph(
            coordinates=coordinates,
            method=method,
            cell_types=cell_types,
            radius=radius,
            k=k,
            mutual_knn=mutual_knn,
            max_edge_length=max_edge_length,
        )
        
        # Store parameters
        method_str = method.value if isinstance(method, GraphMethod) else method
        params = {
            'radius': radius,
            'k': k,
            'mutual_knn': mutual_knn,
            'max_edge_length': max_edge_length,
        }
        
        return cls(G, cell_types, coordinates, method=method_str, params=params)
    
    @classmethod
    def from_coordinates(
        cls,
        coordinates: np.ndarray,
        cell_types: np.ndarray,
        method: Union[str, GraphMethod] = 'proximity',
        **kwargs
    ) -> 'CellGraph':
        """
        Create a CellGraph directly from coordinates and cell types.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Cell coordinates, shape (n_cells, n_dims).
        cell_types : np.ndarray
            Cell type labels.
        method : str or GraphMethod
            Graph construction method.
        **kwargs
            Additional parameters for graph construction.
        
        Returns
        -------
        CellGraph
            Cell graph.
        """
        G = build_graph(
            coordinates=coordinates,
            method=method,
            cell_types=cell_types,
            **kwargs
        )
        
        method_str = method.value if isinstance(method, GraphMethod) else method
        
        return cls(G, cell_types, coordinates, method=method_str, params=kwargs)
    
    @property
    def G(self) -> 'nx.Graph':
        """Access underlying NetworkX graph (alias for backward compatibility)."""
        return self._G

    @G.setter
    def G(self, value: 'nx.Graph'):
        self._G = value

    @property
    def n_nodes(self) -> int:
        """Number of nodes (cells)."""
        return self._G.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self._G.number_of_edges()
    
    @property
    def cell_types(self) -> np.ndarray:
        """Cell type labels."""
        return self._cell_types
    
    @property
    def coordinates(self) -> np.ndarray:
        """Spatial coordinates."""
        return self._coordinates
    
    @property
    def cell_types_unique(self) -> List[str]:
        """List of unique cell types."""
        return list(np.unique(self._cell_types))
    
    @property
    def method(self) -> str:
        """Graph construction method used."""
        return self._method
    
    @property
    def density(self) -> float:
        """Graph density."""
        import networkx as nx
        return nx.density(self.G)
    
    def get_nodes_by_type(self, cell_type: str) -> np.ndarray:
        """
        Get node indices for a specific cell type.
        
        Parameters
        ----------
        cell_type : str
            Cell type to filter by.
        
        Returns
        -------
        np.ndarray
            Array of node indices.
        """
        return np.where(self._cell_types == cell_type)[0]
    
    def subgraph_by_type(
        self,
        cell_types: Union[str, List[str]]
    ) -> 'CellGraph':
        """
        Extract subgraph containing only specified cell types.
        
        Parameters
        ----------
        cell_types : str or list of str
            Cell type(s) to include.
        
        Returns
        -------
        CellGraph
            Subgraph with only the specified cell types.
        """
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        
        # Find nodes to keep
        mask = np.isin(self._cell_types, cell_types)
        nodes_to_keep = np.where(mask)[0]
        
        # Create subgraph
        import networkx as nx
        subG = self.G.subgraph(nodes_to_keep).copy()
        
        # Reindex nodes to be contiguous
        mapping = {old: new for new, old in enumerate(sorted(subG.nodes()))}
        subG = nx.relabel_nodes(subG, mapping)
        
        # Filter cell types and coordinates
        sub_cell_types = self._cell_types[mask]
        sub_coordinates = self._coordinates[mask]
        
        return CellGraph(
            subG, sub_cell_types, sub_coordinates,
            method=self._method, params=self._params
        )
    
    def neighbors_of_type(
        self,
        node: int,
        cell_type: Optional[str] = None
    ) -> List[int]:
        """
        Get neighbors of a node, optionally filtered by type.
        
        Parameters
        ----------
        node : int
            Node index.
        cell_type : str, optional
            Filter neighbors by this cell type.
        
        Returns
        -------
        list of int
            Neighbor node indices.
        """
        neighbors = list(self.G.neighbors(node))
        
        if cell_type is not None:
            neighbors = [n for n in neighbors if self._cell_types[n] == cell_type]
        
        return neighbors
    
    def edge_type_counts(self) -> Dict[Tuple[str, str], int]:
        """
        Count edges by cell type pairs.
        
        Returns
        -------
        dict
            Keys are (type_a, type_b) tuples, values are edge counts.
        """
        counts: Dict[Tuple[str, str], int] = {}
        
        for i, j in self.G.edges():
            type_i = self._cell_types[i]
            type_j = self._cell_types[j]
            
            # Canonical ordering
            key = tuple(sorted([type_i, type_j]))
            counts[key] = counts.get(key, 0) + 1
        
        return counts
    
    def to_networkx(self) -> 'nx.Graph':
        """
        Return the underlying NetworkX graph.
        
        Returns
        -------
        nx.Graph
            Copy of the NetworkX graph.
        """
        return self.G.copy()
    
    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return (
            f"CellGraph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"method={self._method!r})"
        )
    
    def __str__(self) -> str:
        lines = [
            f"CellGraph",
            f"  Nodes: {self.n_nodes}",
            f"  Edges: {self.n_edges}",
            f"  Density: {self.density:.4f}",
            f"  Method: {self._method}",
            f"  Cell types: {len(self.cell_types_unique)}",
        ]
        return '\n'.join(lines)
