"""
Visualization tools for MCP server.

Returns base64-encoded PNG images.

Tools (17 total):
- viz_plot_spatial_scatter: Scatter plot of cells
- viz_plot_cell_types: Cell type colored plot
- viz_plot_density_map: Density heatmap
- viz_plot_marker_expression: Marker expression
- viz_plot_voronoi: Voronoi tessellation
- viz_plot_ripleys_curve: Ripley's H curve
- viz_plot_colocalization_heatmap: CLQ heatmap
- viz_plot_hotspot_map: Gi* hotspot map
- viz_plot_neighborhood_enrichment: Enrichment heatmap
- viz_plot_network: Cell graph on tissue
- viz_plot_degree_distribution: Degree histogram
- viz_plot_mixing_matrix: Type mixing matrix
- viz_plot_topic_composition: LDA topic composition
- viz_plot_topic_spatial: Spatial topic distribution
- viz_plot_mapper_graph: Mapper graph
- viz_plot_trajectory: Time series plot
- viz_save_figure: Save to file
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


class PlotResult(BaseModel):
    """Result containing a plot image."""

    image_base64: str = Field(description="Base64-encoded PNG image")
    width: int
    height: int
    format: str = "png"
    description: str


class SaveResult(BaseModel):
    """Result of saving a figure."""

    file_path: str
    format: str
    success: bool


def register_tools(mcp: "FastMCP") -> None:
    """Register visualization tools with the MCP server."""

    @mcp.tool()
    def viz_plot_spatial_scatter(
        session_id: str,
        data_key: str = "primary",
        color_by: str = "cell_type",
        marker_name: Optional[str] = None,
        figsize_width: float = 8.0,
        figsize_height: float = 8.0,
        point_size: float = 10.0,
        title: Optional[str] = None,
        alpha: float = 0.7,
    ) -> PlotResult:
        """
        Create a spatial scatter plot of cells.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data to plot.
        color_by : str
            How to color: "cell_type" or "marker".
        marker_name : str, optional
            Marker name if color_by="marker".
        figsize_width, figsize_height : float
            Figure dimensions in inches.
        point_size : float
            Size of scatter points.
        title : str, optional
            Plot title.
        alpha : float
            Point transparency.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_spatial_scatter
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

        if color_by == "cell_type":
            plot_spatial_scatter(data, ax=ax, s=point_size, alpha=alpha)
        else:
            plot_spatial_scatter(data, ax=ax, color_by=marker_name, s=point_size, alpha=alpha)

        if title:
            ax.set_title(title)

        fig.tight_layout()
        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Spatial scatter plot of {data.n_cells} cells",
        )

    @mcp.tool()
    def viz_plot_cell_types(
        session_id: str,
        data_key: str = "primary",
        figsize_width: float = 10.0,
        figsize_height: float = 8.0,
        point_size: float = 10.0,
        legend: bool = True,
    ) -> PlotResult:
        """
        Plot cells colored by cell type with legend.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        figsize_width, figsize_height : float
            Figure dimensions.
        point_size : float
            Point size.
        legend : bool
            Whether to show legend.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_cell_types
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_cell_types(data, ax=ax, s=point_size, legend=legend)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Cell type plot with {len(data.cell_types_unique)} types",
        )

    @mcp.tool()
    def viz_plot_density_map(
        session_id: str,
        data_key: str = "primary",
        radius: float = 50.0,
        figsize_width: float = 8.0,
        figsize_height: float = 8.0,
        cmap: str = "viridis",
    ) -> PlotResult:
        """
        Plot cell density heatmap.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        radius : float
            Radius for density calculation.
        figsize_width, figsize_height : float
            Figure dimensions.
        cmap : str
            Colormap name.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_density_map
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_density_map(data, ax=ax, radius=radius, cmap=cmap)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Cell density heatmap",
        )

    @mcp.tool()
    def viz_plot_marker_expression(
        session_id: str,
        marker: str,
        data_key: str = "primary",
        figsize_width: float = 8.0,
        figsize_height: float = 8.0,
        cmap: str = "RdYlBu_r",
        point_size: float = 10.0,
    ) -> PlotResult:
        """
        Plot marker expression spatially.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        marker : str
            Marker name to plot.
        data_key : str
            Key of the data.
        figsize_width, figsize_height : float
            Figure dimensions.
        cmap : str
            Colormap name.
        point_size : float
            Point size.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_marker_expression
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_marker_expression(data, marker=marker, ax=ax, cmap=cmap, s=point_size)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Marker expression: {marker}",
        )

    @mcp.tool()
    def viz_plot_voronoi(
        session_id: str,
        data_key: str = "primary",
        figsize_width: float = 8.0,
        figsize_height: float = 8.0,
        color_by: str = "cell_type",
        alpha: float = 0.5,
    ) -> PlotResult:
        """
        Plot Voronoi tessellation of cells.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        figsize_width, figsize_height : float
            Figure dimensions.
        color_by : str
            Color Voronoi cells by: "cell_type" or "area".
        alpha : float
            Fill transparency.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_voronoi
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_voronoi(data, ax=ax, color_by=color_by, alpha=alpha)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Voronoi tessellation",
        )

    @mcp.tool()
    def viz_plot_ripleys_curve(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
        figsize_width: float = 8.0,
        figsize_height: float = 6.0,
        show_envelope: bool = True,
    ) -> PlotResult:
        """
        Plot Ripley's H curve with CSR envelope.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Cell type to analyze.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radius points.
        figsize_width, figsize_height : float
            Figure dimensions.
        show_envelope : bool
            Show CSR confidence envelope.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_ripleys_curve
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        radii = np.linspace(10, max_radius, n_radii)
        plot_ripleys_curve(data, radii=radii, ax=ax, show_envelope=show_envelope)

        title = "Ripley's H Function"
        if cell_type:
            title += f" ({cell_type})"
        ax.set_title(title)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Ripley's H curve",
        )

    @mcp.tool()
    def viz_plot_colocalization_heatmap(
        session_id: str,
        data_key: str = "primary",
        radius: float = 50.0,
        figsize_width: float = 8.0,
        figsize_height: float = 7.0,
        cmap: str = "RdBu_r",
    ) -> PlotResult:
        """
        Plot colocalization quotient heatmap between all cell types.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        radius : float
            Radius for CLQ computation.
        figsize_width, figsize_height : float
            Figure dimensions.
        cmap : str
            Colormap name.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_colocalization_heatmap
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_colocalization_heatmap(data, ax=ax, radius=radius, cmap=cmap)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Colocalization quotient heatmap",
        )

    @mcp.tool()
    def viz_plot_hotspot_map(
        session_id: str,
        data_key: str = "primary",
        radius: float = 50.0,
        cell_type: Optional[str] = None,
        figsize_width: float = 8.0,
        figsize_height: float = 8.0,
    ) -> PlotResult:
        """
        Plot Getis-Ord Gi* hotspot map.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        radius : float
            Radius for Gi* computation.
        cell_type : str, optional
            Cell type for density hotspots.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_hotspot_map
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_hotspot_map(data, ax=ax, radius=radius, cell_type=cell_type)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Gi* hotspot map",
        )

    @mcp.tool()
    def viz_plot_neighborhood_enrichment(
        session_id: str,
        data_key: str = "primary",
        radius: float = 50.0,
        figsize_width: float = 8.0,
        figsize_height: float = 7.0,
    ) -> PlotResult:
        """
        Plot neighborhood enrichment heatmap.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        radius : float
            Neighborhood radius.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_neighborhood_enrichment
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_neighborhood_enrichment(data, ax=ax, radius=radius)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Neighborhood enrichment heatmap",
        )

    @mcp.tool()
    def viz_plot_network(
        session_id: str,
        graph_key: str = "proximity_graph",
        data_key: str = "primary",
        figsize_width: float = 10.0,
        figsize_height: float = 10.0,
        node_size: float = 5.0,
        edge_alpha: float = 0.3,
    ) -> PlotResult:
        """
        Plot cell graph overlaid on tissue coordinates.

        Parameters
        ----------
        session_id : str
            Session containing graph and data.
        graph_key : str
            Key of the graph.
        data_key : str
            Key of the spatial data.
        figsize_width, figsize_height : float
            Figure dimensions.
        node_size : float
            Size of nodes.
        edge_alpha : float
            Edge transparency.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_cell_graph
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)
        data = session_mgr.load_data(session_id, data_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_cell_graph(graph, data, ax=ax, node_size=node_size, edge_alpha=edge_alpha)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        G = graph._graph if hasattr(graph, "_graph") else graph

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Cell graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        )

    @mcp.tool()
    def viz_plot_degree_distribution(
        session_id: str,
        graph_key: str = "proximity_graph",
        figsize_width: float = 8.0,
        figsize_height: float = 6.0,
        bins: int = 30,
    ) -> PlotResult:
        """
        Plot degree distribution of the graph.

        Parameters
        ----------
        session_id : str
            Session containing the graph.
        graph_key : str
            Key of the graph.
        figsize_width, figsize_height : float
            Figure dimensions.
        bins : int
            Number of histogram bins.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_degree_distribution
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_degree_distribution(graph, ax=ax, bins=bins)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Degree distribution",
        )

    @mcp.tool()
    def viz_plot_mixing_matrix(
        session_id: str,
        graph_key: str = "proximity_graph",
        data_key: str = "primary",
        figsize_width: float = 8.0,
        figsize_height: float = 7.0,
    ) -> PlotResult:
        """
        Plot cell type mixing matrix from graph.

        Parameters
        ----------
        session_id : str
            Session containing graph and data.
        graph_key : str
            Key of the graph.
        data_key : str
            Key of the spatial data.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_mixing_matrix
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        graph = session_mgr.load_graph(session_id, graph_key)
        data = session_mgr.load_data(session_id, data_key)

        if graph is None:
            raise ValueError(f"No graph found with key '{graph_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_mixing_matrix(graph, data, ax=ax)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Cell type mixing matrix",
        )

    @mcp.tool()
    def viz_plot_topic_composition(
        session_id: str,
        model_key: str = "lda_model",
        figsize_width: float = 10.0,
        figsize_height: float = 6.0,
    ) -> PlotResult:
        """
        Plot LDA topic composition (cell types per topic).

        Parameters
        ----------
        session_id : str
            Session containing the LDA model.
        model_key : str
            Key of the LDA model.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_topic_composition
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_topic_composition(model, ax=ax)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="LDA topic composition",
        )

    @mcp.tool()
    def viz_plot_topic_spatial(
        session_id: str,
        model_key: str = "lda_model",
        data_key: str = "primary",
        topic: Optional[int] = None,
        figsize_width: float = 10.0,
        figsize_height: float = 8.0,
    ) -> PlotResult:
        """
        Plot spatial distribution of topic weights.

        Parameters
        ----------
        session_id : str
            Session containing model and data.
        model_key : str
            Key of the LDA model.
        data_key : str
            Key of the spatial data.
        topic : int, optional
            Specific topic to show. If None, shows dominant topic.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_topic_spatial
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_topic_spatial(model, data, topic=topic, ax=ax)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Spatial topic distribution{f' (topic {topic})' if topic is not None else ''}",
        )

    @mcp.tool()
    def viz_plot_mapper_graph(
        session_id: str,
        model_key: str = "mapper_result",
        figsize_width: float = 10.0,
        figsize_height: float = 10.0,
        node_size_scale: float = 100.0,
    ) -> PlotResult:
        """
        Plot Mapper graph visualization.

        Parameters
        ----------
        session_id : str
            Session containing the Mapper result.
        model_key : str
            Key of the Mapper result.
        figsize_width, figsize_height : float
            Figure dimensions.
        node_size_scale : float
            Scale factor for node sizes.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_mapper_graph
        from ..server import get_session_manager
        from ..serialization import figure_to_base64

        session_mgr = get_session_manager()
        result = session_mgr.load_model(session_id, model_key)

        if result is None:
            raise ValueError(f"No Mapper result found with key '{model_key}'")

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
        plot_mapper_graph(result, ax=ax, node_size_scale=node_size_scale)
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description=f"Mapper graph with {result.n_nodes} nodes",
        )

    @mcp.tool()
    def viz_plot_trajectory(
        session_id: str,
        simulation_key: str = "simulation",
        metrics: Optional[List[str]] = None,
        figsize_width: float = 10.0,
        figsize_height: float = 6.0,
    ) -> PlotResult:
        """
        Plot time series trajectories from simulation.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        simulation_key : str
            Key of the simulation.
        metrics : list of str, optional
            Specific metrics to plot. Default: cell counts.
        figsize_width, figsize_height : float
            Figure dimensions.

        Returns
        -------
        PlotResult
            Base64-encoded PNG image.
        """
        from spatialtissuepy.viz import plot_trajectory
        from ..server import get_session_manager
        from ..serialization import figure_to_base64
        import pickle

        session_mgr = get_session_manager()

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        trajectory = sim.cell_count_trajectory()

        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

        cols = metrics if metrics else list(trajectory.columns)
        for col in cols:
            if col in trajectory.columns:
                ax.plot(trajectory.index, trajectory[col], label=col)

        ax.set_xlabel("Time")
        ax.set_ylabel("Cell Count")
        ax.legend()
        ax.set_title("Cell Population Trajectory")
        fig.tight_layout()

        img_base64 = figure_to_base64(fig)
        plt.close(fig)

        return PlotResult(
            image_base64=img_base64,
            width=int(figsize_width * 150),
            height=int(figsize_height * 150),
            description="Simulation trajectory",
        )

    @mcp.tool()
    def viz_save_figure(
        session_id: str,
        output_path: str,
        data_key: str = "primary",
        plot_type: str = "spatial_scatter",
        format: str = "png",
        dpi: int = 300,
    ) -> SaveResult:
        """
        Save a figure to file.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        output_path : str
            Output file path.
        data_key : str
            Key of the data.
        plot_type : str
            Type of plot to generate and save.
        format : str
            Output format: "png", "pdf", "svg".
        dpi : int
            Resolution for raster formats.

        Returns
        -------
        SaveResult
            File path and success status.
        """
        from spatialtissuepy.viz import plot_spatial_scatter
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        fig, ax = plt.subplots(figsize=(8, 8))

        if plot_type == "spatial_scatter":
            plot_spatial_scatter(data, ax=ax)
        else:
            plot_spatial_scatter(data, ax=ax)

        fig.tight_layout()

        path = resolve_data_path(output_path)
        fig.savefig(str(path), format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return SaveResult(
            file_path=str(path),
            format=format,
            success=True,
        )
