"""
Network tools for MCP server.

Graph-based spatial analysis tools.

Tools (14 total):
- network_build_proximity_graph: Radius-based graph
- network_build_knn_graph: k-NN graph
- network_build_delaunay_graph: Delaunay triangulation
- network_build_gabriel_graph: Gabriel graph
- network_degree_centrality: Degree centrality
- network_betweenness_centrality: Betweenness centrality
- network_closeness_centrality: Closeness centrality
- network_eigenvector_centrality: Eigenvector centrality
- network_clustering_coefficient: Local clustering
- network_average_clustering: Global clustering
- network_type_assortativity: Cell type assortativity
- network_degree_assortativity: Degree assortativity
- network_attribute_mixing_matrix: Type mixing matrix
- network_connected_components: Component analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


# --- Pydantic Models ---


class GraphInfo(BaseModel):
    """Information about a constructed graph."""

    session_id: str
    data_key: str
    graph_key: str
    method: str
    n_nodes: int
    n_edges: int
    density: float = Field(description="Edge density (0-1)")
    mean_degree: float
    max_degree: int
    is_connected: bool
    n_components: int


class CentralityResult(BaseModel):
    """Result of centrality computation."""

    session_id: str
    graph_key: str
    metric: str
    n_nodes: int
    mean: float
    std: float
    min: float
    max: float
    top_nodes: List[Dict[str, Any]] = Field(description="Top 10 nodes by centrality")


class ClusteringResult(BaseModel):
    """Result of clustering coefficient computation."""

    session_id: str
    graph_key: str
    global_clustering: float = Field(description="Average clustering coefficient")
    transitivity: float = Field(description="Graph transitivity")
    n_triangles: int


class AssortativityResult(BaseModel):
    """Result of assortativity computation."""

    session_id: str
    graph_key: str
    assortativity: float = Field(description="Assortativity coefficient (-1 to 1)")
    attribute: str
    interpretation: str


class MixingMatrixResult(BaseModel):
    """Result of attribute mixing matrix."""

    session_id: str
    graph_key: str
    attribute: str
    categories: List[str]
    matrix: List[List[float]] = Field(description="Mixing matrix values")
    normalized: bool


class ComponentsResult(BaseModel):
    """Result of connected components analysis."""

    session_id: str
    graph_key: str
    n_components: int
    largest_component_size: int
    largest_component_fraction: float
    component_sizes: List[int]


# --- Helper Functions ---


def _compute_clustering(session_id: str, graph_key: str) -> ClusteringResult:
    """Internal helper to compute clustering coefficients."""
    import networkx as nx
    from ..server import get_session_manager

    session_mgr = get_session_manager()
    graph = session_mgr.load_graph(session_id, graph_key)

    if graph is None:
        raise ValueError(f"No graph found with key '{graph_key}'")

    G = graph._graph if hasattr(graph, "_graph") else graph

    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    n_triangles = sum(nx.triangles(G).values()) // 3

    return ClusteringResult(
        session_id=session_id,
        graph_key=graph_key,
        global_clustering=float(avg_clustering),
        transitivity=float(transitivity),
        n_triangles=n_triangles,
    )


# --- Tool Registration ---


def register_tools(mcp: "FastMCP") -> None:
    """Register network tools with the MCP server."""

    @mcp.tool()
    def network_build_proximity_graph(
        session_id: str,
        radius: float,
        data_key: str = "primary",
        graph_key: str = "proximity_graph",
    ) -> GraphInfo:
        """
        Build a proximity graph connecting cells within a radius.

        Cells within the specified distance are connected by edges.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        radius : float
            Maximum distance for edge connection.
        data_key : str
            Key of the spatial data.
        graph_key : str
            Key to store the graph.

        Returns
        -------
        GraphInfo
            Information about the constructed graph.
        """
        from spatialtissuepy.network import build_proximity_graph
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        graph = build_proximity_graph(
            data.coordinates,
            radius=radius,
            cell_types=data.cell_types,
        )

        # Store graph
        session_mgr.store_graph(session_id, graph_key, graph, {"method": "proximity", "radius": radius})

        import networkx as nx
        G = graph._graph if hasattr(graph, "_graph") else graph

        degrees = [d for _, d in G.degree()]

        return GraphInfo(
            session_id=session_id,
            data_key=data_key,
            graph_key=graph_key,
            method="proximity",
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            density=nx.density(G),
            mean_degree=float(np.mean(degrees)),
            max_degree=int(np.max(degrees)),
            is_connected=nx.is_connected(G),
            n_components=nx.number_connected_components(G),
        )

    @mcp.tool()
    def network_build_knn_graph(
        session_id: str,
        k: int = 5,
        data_key: str = "primary",
        graph_key: str = "knn_graph",
        mutual: bool = False,
    ) -> GraphInfo:
        """
        Build a k-nearest neighbors graph.

        Each cell is connected to its k nearest neighbors.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        k : int
            Number of neighbors per cell.
        data_key : str
            Key of the spatial data.
        graph_key : str
            Key to store the graph.
        mutual : bool
            If True, only keep mutual neighbor connections.

        Returns
        -------
        GraphInfo
            Information about the constructed graph.
        """
        from spatialtissuepy.network import build_knn_graph
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        graph = build_knn_graph(
            data.coordinates,
            k=k,
            cell_types=data.cell_types,
            mutual=mutual,
        )

        session_mgr.store_graph(session_id, graph_key, graph, {"method": "knn", "k": k, "mutual": mutual})

        import networkx as nx
        G = graph._graph if hasattr(graph, "_graph") else graph

        degrees = [d for _, d in G.degree()]

        return GraphInfo(
            session_id=session_id,
            data_key=data_key,
            graph_key=graph_key,
            method="knn",
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            density=nx.density(G),
            mean_degree=float(np.mean(degrees)),
            max_degree=int(np.max(degrees)),
            is_connected=nx.is_connected(G),
            n_components=nx.number_connected_components(G),
        )

    @mcp.tool()
    def network_build_delaunay_graph(
        session_id: str,
        data_key: str = "primary",
        graph_key: str = "delaunay_graph",
    ) -> GraphInfo:
        """
        Build a Delaunay triangulation graph.

        Creates a planar graph based on Delaunay triangulation of cell positions.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        graph_key : str
            Key to store the graph.

        Returns
        -------
        GraphInfo
            Information about the constructed graph.
        """
        from spatialtissuepy.network import build_delaunay_graph
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        graph = build_delaunay_graph(
            data.coordinates,
            cell_types=data.cell_types,
        )

        session_mgr.store_graph(session_id, graph_key, graph, {"method": "delaunay"})

        import networkx as nx
        G = graph._graph if hasattr(graph, "_graph") else graph

        degrees = [d for _, d in G.degree()]

        return GraphInfo(
            session_id=session_id,
            data_key=data_key,
            graph_key=graph_key,
            method="delaunay",
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            density=nx.density(G),
            mean_degree=float(np.mean(degrees)),
            max_degree=int(np.max(degrees)),
            is_connected=nx.is_connected(G),
            n_components=nx.number_connected_components(G),
        )

    @mcp.tool()
    def network_build_gabriel_graph(
        session_id: str,
        data_key: str = "primary",
        graph_key: str = "gabriel_graph",
    ) -> GraphInfo:
        """
        Build a Gabriel graph.

        A subgraph of Delaunay where edges only exist if no other point
        lies within the diametric circle of the edge endpoints.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        graph_key : str
            Key to store the graph.

        Returns
        -------
        GraphInfo
            Information about the constructed graph.
        """
        from spatialtissuepy.network import build_gabriel_graph
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        graph = build_gabriel_graph(
            data.coordinates,
            cell_types=data.cell_types,
        )

        session_mgr.store_graph(session_id, graph_key, graph, {"method": "gabriel"})

        import networkx as nx
        G = graph._graph if hasattr(graph, "_graph") else graph

        degrees = [d for _, d in G.degree()]

        return GraphInfo(
            session_id=session_id,
            data_key=data_key,
            graph_key=graph_key,
            method="gabriel",
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            density=nx.density(G),
            mean_degree=float(np.mean(degrees)),
            max_degree=int(np.max(degrees)),
            is_connected=nx.is_connected(G),
            n_components=nx.number_connected_components(G),
        )

    @mcp.tool()
    def network_degree_centrality(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> CentralityResult:
        """
        Compute degree centrality for all nodes.

        Degree centrality is the fraction of nodes each node is connected to.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph to analyze.

        Returns
        -------
        CentralityResult
            Centrality statistics and top nodes.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph
        centrality = nx.degree_centrality(G)
        values = list(centrality.values())

        # Top 10 nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [{"node": n, "centrality": float(c)} for n, c in sorted_nodes]

        return CentralityResult(
            session_id=session_id,
            graph_key=graph_key,
            metric="degree_centrality",
            n_nodes=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            top_nodes=top_nodes,
        )

    @mcp.tool()
    def network_betweenness_centrality(
        session_id: str,
        graph_key: str = "proximity_graph",
        k: Optional[int] = None,
    ) -> CentralityResult:
        """
        Compute betweenness centrality for all nodes.

        Betweenness measures how often a node lies on shortest paths.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.
        k : int, optional
            Sample size for approximation (for large graphs).

        Returns
        -------
        CentralityResult
            Centrality statistics and top nodes.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph
        centrality = nx.betweenness_centrality(G, k=k)
        values = list(centrality.values())

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [{"node": n, "centrality": float(c)} for n, c in sorted_nodes]

        return CentralityResult(
            session_id=session_id,
            graph_key=graph_key,
            metric="betweenness_centrality",
            n_nodes=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            top_nodes=top_nodes,
        )

    @mcp.tool()
    def network_closeness_centrality(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> CentralityResult:
        """
        Compute closeness centrality for all nodes.

        Closeness is the inverse of the average shortest path to all other nodes.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.

        Returns
        -------
        CentralityResult
            Centrality statistics and top nodes.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph
        centrality = nx.closeness_centrality(G)
        values = list(centrality.values())

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [{"node": n, "centrality": float(c)} for n, c in sorted_nodes]

        return CentralityResult(
            session_id=session_id,
            graph_key=graph_key,
            metric="closeness_centrality",
            n_nodes=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            top_nodes=top_nodes,
        )

    @mcp.tool()
    def network_eigenvector_centrality(
        session_id: str,
        graph_key: str = "proximity_graph",
        max_iter: int = 100,
    ) -> CentralityResult:
        """
        Compute eigenvector centrality for all nodes.

        A node is important if it's connected to other important nodes.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.
        max_iter : int
            Maximum iterations for convergence.

        Returns
        -------
        CentralityResult
            Centrality statistics and top nodes.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph

        try:
            centrality = nx.eigenvector_centrality(G, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.eigenvector_centrality_numpy(G)

        values = list(centrality.values())

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [{"node": n, "centrality": float(c)} for n, c in sorted_nodes]

        return CentralityResult(
            session_id=session_id,
            graph_key=graph_key,
            metric="eigenvector_centrality",
            n_nodes=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            top_nodes=top_nodes,
        )

    @mcp.tool()
    def network_clustering_coefficient(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> ClusteringResult:
        """
        Compute clustering coefficients for the graph.

        Measures tendency of nodes to cluster together (form triangles).

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.

        Returns
        -------
        ClusteringResult
            Global and local clustering statistics.
        """
        return _compute_clustering(session_id, graph_key)

    @mcp.tool()
    def network_average_clustering(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> ClusteringResult:
        """
        Compute average clustering coefficient.

        Alias for network_clustering_coefficient.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.

        Returns
        -------
        ClusteringResult
            Clustering statistics.
        """
        return _compute_clustering(session_id, graph_key)

    @mcp.tool()
    def network_type_assortativity(
        session_id: str,
        graph_key: str = "proximity_graph",
        data_key: str = "primary",
    ) -> AssortativityResult:
        """
        Compute assortativity by cell type.

        Measures tendency of cells to connect to same or different types.
        - Positive: Cells prefer connecting to same type (homophily)
        - Zero: Random mixing
        - Negative: Cells prefer connecting to different types (heterophily)

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.
        data_key : str
            Key of spatial data (for cell type labels).

        Returns
        -------
        AssortativityResult
            Assortativity coefficient and interpretation.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)
        data = session_mgr.load_data(session_id, data_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph

        # Add cell type as node attribute
        for i, cell_type in enumerate(data.cell_types):
            if i in G.nodes:
                G.nodes[i]["cell_type"] = cell_type

        assortativity = nx.attribute_assortativity_coefficient(G, "cell_type")

        if assortativity > 0.3:
            interpretation = "Strong homophily: cells preferentially connect to same type"
        elif assortativity > 0.1:
            interpretation = "Moderate homophily"
        elif assortativity < -0.3:
            interpretation = "Strong heterophily: cells preferentially connect to different types"
        elif assortativity < -0.1:
            interpretation = "Moderate heterophily"
        else:
            interpretation = "Random mixing between cell types"

        return AssortativityResult(
            session_id=session_id,
            graph_key=graph_key,
            assortativity=float(assortativity),
            attribute="cell_type",
            interpretation=interpretation,
        )

    @mcp.tool()
    def network_degree_assortativity(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> AssortativityResult:
        """
        Compute degree assortativity.

        Measures tendency of high-degree nodes to connect to other high-degree nodes.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.

        Returns
        -------
        AssortativityResult
            Degree assortativity coefficient.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph

        assortativity = nx.degree_assortativity_coefficient(G)

        if assortativity > 0.3:
            interpretation = "Assortative: hubs connect to hubs"
        elif assortativity > 0:
            interpretation = "Mildly assortative"
        elif assortativity < -0.3:
            interpretation = "Disassortative: hubs connect to low-degree nodes"
        elif assortativity < 0:
            interpretation = "Mildly disassortative"
        else:
            interpretation = "Neutral degree mixing"

        return AssortativityResult(
            session_id=session_id,
            graph_key=graph_key,
            assortativity=float(assortativity),
            attribute="degree",
            interpretation=interpretation,
        )

    @mcp.tool()
    def network_attribute_mixing_matrix(
        session_id: str,
        graph_key: str = "proximity_graph",
        data_key: str = "primary",
        normalized: bool = True,
    ) -> MixingMatrixResult:
        """
        Compute cell type mixing matrix.

        Shows the fraction of edges between each pair of cell types.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.
        data_key : str
            Key of spatial data.
        normalized : bool
            If True, normalize to fractions.

        Returns
        -------
        MixingMatrixResult
            Mixing matrix between cell types.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)
        data = session_mgr.load_data(session_id, data_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph

        # Add cell type as node attribute
        for i, cell_type in enumerate(data.cell_types):
            if i in G.nodes:
                G.nodes[i]["cell_type"] = cell_type

        mixing = nx.attribute_mixing_matrix(G, "cell_type", normalized=normalized)
        categories = list(data.cell_types_unique)

        return MixingMatrixResult(
            session_id=session_id,
            graph_key=graph_key,
            attribute="cell_type",
            categories=categories,
            matrix=mixing.tolist(),
            normalized=normalized,
        )

    @mcp.tool()
    def network_connected_components(
        session_id: str,
        graph_key: str = "proximity_graph",
    ) -> ComponentsResult:
        """
        Analyze connected components in the graph.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.

        Returns
        -------
        ComponentsResult
            Component count and sizes.
        """
        import networkx as nx
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        G = graph._graph if hasattr(graph, "_graph") else graph

        components = list(nx.connected_components(G))
        sizes = sorted([len(c) for c in components], reverse=True)

        largest = sizes[0] if sizes else 0
        total = G.number_of_nodes()

        return ComponentsResult(
            session_id=session_id,
            graph_key=graph_key,
            n_components=len(components),
            largest_component_size=largest,
            largest_component_fraction=largest / total if total > 0 else 0,
            component_sizes=sizes[:20],  # Top 20 component sizes
        )
