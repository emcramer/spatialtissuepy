"""
Serialization utilities for MCP server.

Handles conversion of numpy arrays, pandas DataFrames, NetworkX graphs,
and model objects to/from JSON-serializable formats.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.lda import SpatialLDA
    from spatialtissuepy.topology import MapperResult
    from spatialtissuepy.network import CellGraph


class MCPSerializer:
    """
    Serialization utilities for MCP responses.

    Provides methods to convert numpy arrays, pandas DataFrames,
    and other complex objects to JSON-serializable formats.

    Examples
    --------
    >>> serializer = MCPSerializer()
    >>> arr = np.array([1, 2, 3])
    >>> json_data = serializer.numpy_to_json(arr)
    >>> restored = serializer.json_to_numpy(json_data)
    """

    def numpy_to_json(
        self,
        arr: np.ndarray,
        max_size: int = 10000,
    ) -> Dict[str, Any]:
        """
        Convert numpy array to JSON-serializable dict.

        Parameters
        ----------
        arr : np.ndarray
            Array to convert.
        max_size : int
            Maximum array size for full serialization.
            Larger arrays return summary only.

        Returns
        -------
        dict
            JSON-serializable representation.
        """
        if arr.size > max_size:
            return {
                "_type": "ndarray_summary",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                "max": float(np.nanmax(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                "mean": float(np.nanmean(arr)) if np.issubdtype(arr.dtype, np.number) else None,
                "std": float(np.nanstd(arr)) if np.issubdtype(arr.dtype, np.number) else None,
            }
        return {
            "_type": "ndarray",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tolist(),
        }

    def json_to_numpy(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert JSON dict back to numpy array."""
        if data.get("_type") == "ndarray_summary":
            raise ValueError("Cannot reconstruct array from summary")
        return np.array(data["data"], dtype=data["dtype"])

    def dataframe_to_json(
        self,
        df: pd.DataFrame,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """
        Convert DataFrame to JSON-serializable dict.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert.
        max_rows : int
            Maximum rows for full serialization.
            Larger DataFrames return head + summary.

        Returns
        -------
        dict
            JSON-serializable representation.
        """
        if len(df) > max_rows:
            return {
                "_type": "dataframe_summary",
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head": {
                    "columns": list(df.columns),
                    "index": list(df.head(10).index),
                    "data": df.head(10).values.tolist(),
                },
                "describe": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None,
            }
        return {
            "_type": "dataframe",
            "columns": list(df.columns),
            "index": [str(i) for i in df.index],  # Convert index to strings
            "data": df.values.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    def json_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert JSON dict back to DataFrame."""
        if data.get("_type") == "dataframe_summary":
            # Return just the head for summaries
            head_data = data["head"]
            return pd.DataFrame(
                head_data["data"],
                columns=head_data["columns"],
                index=head_data["index"],
            )
        return pd.DataFrame(
            data["data"],
            columns=data["columns"],
            index=data["index"],
        )

    def series_to_json(self, series: pd.Series) -> Dict[str, Any]:
        """Convert Series to JSON-serializable dict."""
        return {
            "_type": "series",
            "name": series.name,
            "index": [str(i) for i in series.index],
            "data": series.tolist(),
            "dtype": str(series.dtype),
        }

    def json_to_series(self, data: Dict[str, Any]) -> pd.Series:
        """Convert JSON dict back to Series."""
        return pd.Series(
            data["data"],
            index=data["index"],
            name=data["name"],
            dtype=data.get("dtype"),
        )

    def serialize_result(self, obj: Any) -> Any:
        """
        Serialize any supported object type for MCP response.

        Parameters
        ----------
        obj : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-serializable representation.
        """
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            return self.numpy_to_json(obj)

        elif isinstance(obj, pd.DataFrame):
            return self.dataframe_to_json(obj)

        elif isinstance(obj, pd.Series):
            return self.series_to_json(obj)

        elif isinstance(obj, dict):
            return {k: self.serialize_result(v) for k, v in obj.items()}

        elif isinstance(obj, (list, tuple)):
            return [self.serialize_result(item) for item in obj]

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, (int, float, str, bool)):
            return obj

        elif hasattr(obj, "to_dict"):
            # Objects with to_dict method (e.g., Pydantic models)
            return obj.to_dict()

        elif hasattr(obj, "__dict__"):
            # Generic object serialization
            return {
                "_type": type(obj).__name__,
                "attrs": {
                    k: self.serialize_result(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                },
            }

        return str(obj)


def serialize_graph(
    graph: "CellGraph",
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Serialize a NetworkX graph for exact reconstruction.

    Stores nodes, edges, and all attributes separately.

    Parameters
    ----------
    graph : CellGraph
        Graph to serialize.
    params : dict, optional
        Construction parameters to store.

    Returns
    -------
    dict
        JSON-serializable graph representation.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for graph serialization")

    # Get underlying NetworkX graph
    G = graph._graph if hasattr(graph, "_graph") else graph

    # Serialize nodes with attributes
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        node_data = {"id": int(node_id) if isinstance(node_id, (int, np.integer)) else node_id}
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                node_data[k] = v.tolist()
            elif isinstance(v, (tuple, set)):
                node_data[k] = list(v)
            elif isinstance(v, (np.integer, np.floating)):
                node_data[k] = float(v)
            else:
                node_data[k] = v
        nodes.append(node_data)

    # Serialize edges with attributes
    edges = []
    for u, v, attrs in G.edges(data=True):
        edge_data = {
            "source": int(u) if isinstance(u, (int, np.integer)) else u,
            "target": int(v) if isinstance(v, (int, np.integer)) else v,
        }
        for k, val in attrs.items():
            if isinstance(val, (np.integer, np.floating)):
                edge_data[k] = float(val)
            elif isinstance(val, np.ndarray):
                edge_data[k] = val.tolist()
            else:
                edge_data[k] = val
        edges.append(edge_data)

    # Graph-level attributes
    graph_attrs = {}
    for k, v in G.graph.items():
        if isinstance(v, (np.integer, np.floating)):
            graph_attrs[k] = float(v)
        elif isinstance(v, np.ndarray):
            graph_attrs[k] = v.tolist()
        else:
            graph_attrs[k] = v

    return {
        "_type": "graph",
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
        "graph_attrs": graph_attrs,
        "params": params or {},
    }


def deserialize_graph(data: Dict[str, Any]) -> "CellGraph":
    """
    Reconstruct NetworkX graph from serialized data.

    Parameters
    ----------
    data : dict
        Serialized graph data.

    Returns
    -------
    CellGraph
        Reconstructed graph.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for graph deserialization")

    from spatialtissuepy.network import CellGraph

    G = nx.Graph()
    G.graph.update(data.get("graph_attrs", {}))

    for node in data["nodes"]:
        node_copy = node.copy()
        node_id = node_copy.pop("id")
        G.add_node(node_id, **node_copy)

    for edge in data["edges"]:
        edge_copy = edge.copy()
        source = edge_copy.pop("source")
        target = edge_copy.pop("target")
        G.add_edge(source, target, **edge_copy)

    # Create CellGraph wrapper
    cell_graph = CellGraph.__new__(CellGraph)
    cell_graph._graph = G
    if hasattr(cell_graph, "_construction_params"):
        cell_graph._construction_params = data.get("params", {})

    return cell_graph


def serialize_model(
    model: Union["SpatialLDA", "MapperResult", Any],
    model_type: str,
) -> Dict[str, Any]:
    """
    Serialize LDA or Mapper model for reproducibility.

    Parameters
    ----------
    model : SpatialLDA or MapperResult
        Model to serialize.
    model_type : str
        Type identifier ("lda" or "mapper").

    Returns
    -------
    dict
        JSON-serializable model representation.
    """
    if model_type == "lda":
        result = {
            "_type": "spatial_lda",
            "n_topics": getattr(model, "n_topics", None),
            "neighborhood_radius": getattr(model, "neighborhood_radius", None),
            "neighborhood_method": getattr(model, "neighborhood_method", "radius"),
            "random_state": getattr(model, "random_state", None),
        }

        # Store fitted components if available
        if hasattr(model, "components_") and model.components_ is not None:
            result["components"] = model.components_.tolist()
        if hasattr(model, "cell_types_") and model.cell_types_ is not None:
            result["cell_types"] = list(model.cell_types_)

        return result

    elif model_type == "mapper":
        nodes_data = []
        if hasattr(model, "nodes"):
            for n in model.nodes:
                node_info = {
                    "id": getattr(n, "node_id", None),
                    "size": getattr(n, "size", len(getattr(n, "cells", []))),
                }
                if hasattr(n, "cells"):
                    node_info["cells"] = list(n.cells)
                nodes_data.append(node_info)

        edges_data = []
        if hasattr(model, "edges"):
            for e in model.edges:
                edges_data.append({
                    "source": getattr(e, "source", None),
                    "target": getattr(e, "target", None),
                })

        return {
            "_type": "mapper_result",
            "n_nodes": getattr(model, "n_nodes", len(nodes_data)),
            "n_edges": getattr(model, "n_edges", len(edges_data)),
            "n_components": getattr(model, "n_components", None),
            "nodes": nodes_data,
            "edges": edges_data,
            "params": getattr(model, "params", {}),
        }

    # Generic fallback
    return {
        "_type": model_type,
        "data": str(model),
    }


def deserialize_model(
    data: Dict[str, Any],
) -> Union["SpatialLDA", "MapperResult", Dict]:
    """
    Reconstruct model from serialized data.

    Parameters
    ----------
    data : dict
        Serialized model data.

    Returns
    -------
    SpatialLDA, MapperResult, or dict
        Reconstructed model.
    """
    model_type = data.get("_type")

    if model_type == "spatial_lda":
        try:
            from spatialtissuepy.lda import SpatialLDA

            model = SpatialLDA(
                n_topics=data["n_topics"],
                neighborhood_radius=data.get("neighborhood_radius", 50.0),
                neighborhood_method=data.get("neighborhood_method", "radius"),
                random_state=data.get("random_state"),
            )

            if data.get("components"):
                model.components_ = np.array(data["components"])
            if data.get("cell_types"):
                model.cell_types_ = np.array(data["cell_types"])

            return model
        except ImportError:
            return data

    elif model_type == "mapper_result":
        try:
            from spatialtissuepy.topology import MapperResult, MapperNode, MapperEdge

            nodes = []
            for n in data.get("nodes", []):
                node = MapperNode(
                    node_id=n["id"],
                    cells=set(n.get("cells", [])),
                )
                nodes.append(node)

            edges = []
            for e in data.get("edges", []):
                edge = MapperEdge(
                    source=e["source"],
                    target=e["target"],
                )
                edges.append(edge)

            return MapperResult(
                nodes=nodes,
                edges=edges,
                params=data.get("params", {}),
            )
        except (ImportError, TypeError):
            return data

    return data


def figure_to_base64(fig, dpi: int = 150, format: str = "png") -> str:
    """
    Convert matplotlib figure to base64-encoded string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert.
    dpi : int
        Resolution in dots per inch.
    format : str
        Image format (png, jpg, svg).

    Returns
    -------
    str
        Base64-encoded image string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def base64_to_figure(b64_string: str):
    """
    Convert base64 string back to figure (for testing).

    Parameters
    ----------
    b64_string : str
        Base64-encoded image.

    Returns
    -------
    PIL.Image
        Decoded image.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required for base64 to image conversion")

    buf = io.BytesIO(base64.b64decode(b64_string))
    return Image.open(buf)
