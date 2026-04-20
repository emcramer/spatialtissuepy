"""
Topology tools for MCP server.

Mapper/TDA analysis tools.

Tools (10 total):
- topology_run_mapper: Run Mapper algorithm
- topology_density_filter: Density filter
- topology_eccentricity_filter: Eccentricity filter
- topology_pca_filter: PCA projection filter
- topology_distance_to_type_filter: Distance to cell type
- topology_radial_filter: Radial distance filter
- topology_get_nodes: Node information
- topology_get_edges: Edge information
- topology_get_components: Connected components
- topology_hub_nodes: Identify hub nodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


class MapperResult(BaseModel):
    """Result of Mapper computation."""

    session_id: str
    data_key: str
    model_key: str
    n_nodes: int
    n_edges: int
    n_components: int
    filter_function: str
    n_intervals: int
    overlap: float


class MapperNodesResult(BaseModel):
    """Node information from Mapper."""

    session_id: str
    model_key: str
    n_nodes: int
    nodes: List[Dict[str, Any]] = Field(description="Node ID, size, and composition")


class MapperEdgesResult(BaseModel):
    """Edge information from Mapper."""

    session_id: str
    model_key: str
    n_edges: int
    edges: List[Dict[str, Any]] = Field(description="Source, target, weight")


class MapperComponentsResult(BaseModel):
    """Connected components in Mapper graph."""

    session_id: str
    model_key: str
    n_components: int
    component_sizes: List[int]


class HubNodesResult(BaseModel):
    """Hub nodes in Mapper graph."""

    session_id: str
    model_key: str
    n_hubs: int
    hubs: List[Dict[str, Any]] = Field(description="Hub node ID, degree, cells")


class FilterResult(BaseModel):
    """Result of computing a filter function."""

    session_id: str
    data_key: str
    filter_name: str
    n_cells: int
    min_value: float
    max_value: float
    mean_value: float


def register_tools(mcp: "FastMCP") -> None:
    """Register topology tools with the MCP server."""

    @mcp.tool()
    def topology_run_mapper(
        session_id: str,
        data_key: str = "primary",
        model_key: str = "mapper_result",
        filter_function: str = "density",
        n_intervals: int = 10,
        overlap: float = 0.4,
        neighborhood_radius: float = 50.0,
        clustering_method: str = "dbscan",
        min_cluster_size: int = 3,
    ) -> MapperResult:
        """
        Run the Mapper algorithm for topological data analysis.

        Mapper creates a simplified graph representation of the data
        that preserves topological features.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        model_key : str
            Key to store the Mapper result.
        filter_function : str
            Filter function: "density", "eccentricity", "pca", "x", "y".
        n_intervals : int
            Number of intervals in the cover.
        overlap : float
            Overlap fraction between intervals (0-1).
        neighborhood_radius : float
            Radius for density filter.
        clustering_method : str
            Clustering method: "dbscan", "agglomerative".
        min_cluster_size : int
            Minimum cells per cluster.

        Returns
        -------
        MapperResult
            Information about the Mapper graph.
        """
        from spatialtissuepy.topology import spatial_mapper
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        result = spatial_mapper(
            data,
            filter_fn=filter_function,
            n_intervals=n_intervals,
            overlap=overlap,
            neighborhood_radius=neighborhood_radius,
            clustering=clustering_method,
            min_cluster_size=min_cluster_size,
        )

        session_mgr.store_model(session_id, model_key, result, "mapper")

        return MapperResult(
            session_id=session_id,
            data_key=data_key,
            model_key=model_key,
            n_nodes=result.n_nodes,
            n_edges=result.n_edges,
            n_components=result.n_components,
            filter_function=filter_function,
            n_intervals=n_intervals,
            overlap=overlap,
        )

    @mcp.tool()
    def topology_density_filter(
        session_id: str,
        data_key: str = "primary",
        radius: float = 50.0,
    ) -> FilterResult:
        """
        Compute density filter function.

        Assigns each cell a value based on local density.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        radius : float
            Radius for density calculation.

        Returns
        -------
        FilterResult
            Statistics of filter values.
        """
        from spatialtissuepy.topology import density_filter
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        values = density_filter(data, radius=radius)

        return FilterResult(
            session_id=session_id,
            data_key=data_key,
            filter_name="density",
            n_cells=len(values),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            mean_value=float(np.mean(values)),
        )

    @mcp.tool()
    def topology_eccentricity_filter(
        session_id: str,
        data_key: str = "primary",
    ) -> FilterResult:
        """
        Compute eccentricity filter function.

        Eccentricity is the maximum distance to any other point.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.

        Returns
        -------
        FilterResult
            Statistics of filter values.
        """
        from spatialtissuepy.topology import eccentricity_filter
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        values = eccentricity_filter(data)

        return FilterResult(
            session_id=session_id,
            data_key=data_key,
            filter_name="eccentricity",
            n_cells=len(values),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            mean_value=float(np.mean(values)),
        )

    @mcp.tool()
    def topology_pca_filter(
        session_id: str,
        data_key: str = "primary",
        component: int = 0,
    ) -> FilterResult:
        """
        Compute PCA projection filter.

        Projects coordinates onto a principal component.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        component : int
            Which principal component (0=first).

        Returns
        -------
        FilterResult
            Statistics of filter values.
        """
        from spatialtissuepy.topology import pca_filter
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        values = pca_filter(data, component=component)

        return FilterResult(
            session_id=session_id,
            data_key=data_key,
            filter_name=f"pca_{component}",
            n_cells=len(values),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            mean_value=float(np.mean(values)),
        )

    @mcp.tool()
    def topology_distance_to_type_filter(
        session_id: str,
        cell_type: str,
        data_key: str = "primary",
    ) -> FilterResult:
        """
        Compute distance-to-cell-type filter.

        Assigns each cell the distance to the nearest cell of specified type.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        cell_type : str
            Target cell type.
        data_key : str
            Key of the spatial data.

        Returns
        -------
        FilterResult
            Statistics of filter values.
        """
        from spatialtissuepy.topology import distance_to_type_filter
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        values = distance_to_type_filter(data, cell_type=cell_type)

        return FilterResult(
            session_id=session_id,
            data_key=data_key,
            filter_name=f"distance_to_{cell_type}",
            n_cells=len(values),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            mean_value=float(np.mean(values)),
        )

    @mcp.tool()
    def topology_radial_filter(
        session_id: str,
        data_key: str = "primary",
        center: Optional[List[float]] = None,
    ) -> FilterResult:
        """
        Compute radial distance filter from center.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        center : list of float, optional
            Center point [x, y]. Default: centroid.

        Returns
        -------
        FilterResult
            Statistics of filter values.
        """
        from spatialtissuepy.topology import radial_filter
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if center is not None:
            center = np.array(center)

        values = radial_filter(data, center=center)

        return FilterResult(
            session_id=session_id,
            data_key=data_key,
            filter_name="radial",
            n_cells=len(values),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            mean_value=float(np.mean(values)),
        )

    @mcp.tool()
    def topology_get_nodes(
        session_id: str,
        model_key: str = "mapper_result",
        data_key: str = "primary",
    ) -> MapperNodesResult:
        """
        Get node information from Mapper result.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the Mapper result.
        data_key : str
            Key of the spatial data (for cell type info).

        Returns
        -------
        MapperNodesResult
            Node IDs, sizes, and compositions.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        result = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if result is None:
            raise ValueError(f"No Mapper result found with key '{model_key}'")

        nodes_info = []
        for node in result.nodes:
            node_data = {
                "id": node.node_id,
                "size": node.size,
            }
            if data is not None and hasattr(node, "cells"):
                cell_types = [data.cell_types[i] for i in node.cells if i < len(data.cell_types)]
                unique, counts = np.unique(cell_types, return_counts=True)
                node_data["composition"] = {str(t): int(c) for t, c in zip(unique, counts)}
            nodes_info.append(node_data)

        return MapperNodesResult(
            session_id=session_id,
            model_key=model_key,
            n_nodes=len(nodes_info),
            nodes=nodes_info,
        )

    @mcp.tool()
    def topology_get_edges(
        session_id: str,
        model_key: str = "mapper_result",
    ) -> MapperEdgesResult:
        """
        Get edge information from Mapper result.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the Mapper result.

        Returns
        -------
        MapperEdgesResult
            Edge source, target, and weights.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        result = session_mgr.load_model(session_id, model_key)

        if result is None:
            raise ValueError(f"No Mapper result found with key '{model_key}'")

        edges_info = []
        for edge in result.edges:
            edges_info.append({
                "source": edge.source,
                "target": edge.target,
                "weight": getattr(edge, "weight", 1),
            })

        return MapperEdgesResult(
            session_id=session_id,
            model_key=model_key,
            n_edges=len(edges_info),
            edges=edges_info,
        )

    @mcp.tool()
    def topology_get_components(
        session_id: str,
        model_key: str = "mapper_result",
    ) -> MapperComponentsResult:
        """
        Get connected components from Mapper graph.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the Mapper result.

        Returns
        -------
        MapperComponentsResult
            Number and sizes of components.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        result = session_mgr.load_model(session_id, model_key)

        if result is None:
            raise ValueError(f"No Mapper result found with key '{model_key}'")

        # Build graph and find components
        import networkx as nx
        G = nx.Graph()
        for node in result.nodes:
            G.add_node(node.node_id)
        for edge in result.edges:
            G.add_edge(edge.source, edge.target)

        components = list(nx.connected_components(G))
        sizes = sorted([len(c) for c in components], reverse=True)

        return MapperComponentsResult(
            session_id=session_id,
            model_key=model_key,
            n_components=len(components),
            component_sizes=sizes,
        )

    @mcp.tool()
    def topology_hub_nodes(
        session_id: str,
        model_key: str = "mapper_result",
        min_degree: int = 3,
    ) -> HubNodesResult:
        """
        Identify hub nodes in the Mapper graph.

        Hub nodes have high connectivity and may represent
        transitional regions in the data.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the Mapper result.
        min_degree : int
            Minimum degree to be considered a hub.

        Returns
        -------
        HubNodesResult
            Hub node information.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        result = session_mgr.load_model(session_id, model_key)

        if result is None:
            raise ValueError(f"No Mapper result found with key '{model_key}'")

        # Build graph
        import networkx as nx
        G = nx.Graph()
        node_sizes = {}
        for node in result.nodes:
            G.add_node(node.node_id)
            node_sizes[node.node_id] = node.size
        for edge in result.edges:
            G.add_edge(edge.source, edge.target)

        # Find hubs
        hubs = []
        for node_id, degree in G.degree():
            if degree >= min_degree:
                hubs.append({
                    "node_id": node_id,
                    "degree": degree,
                    "n_cells": node_sizes.get(node_id, 0),
                })

        hubs.sort(key=lambda x: x["degree"], reverse=True)

        return HubNodesResult(
            session_id=session_id,
            model_key=model_key,
            n_hubs=len(hubs),
            hubs=hubs[:20],  # Top 20 hubs
        )
