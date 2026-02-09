"""
Data tools for MCP server.

Tools for loading, saving, and inspecting SpatialTissueData.

Tools (14 total):
- data_load_csv: Load spatial data from CSV
- data_load_json: Load spatial data from JSON
- data_save_csv: Export data to CSV
- data_save_json: Export data to JSON
- data_get_info: Get dataset summary
- data_get_cell_types: List unique cell types
- data_get_cell_counts: Get counts by type
- data_get_bounds: Get spatial bounds
- data_get_markers: List marker columns
- data_subset_by_type: Filter by cell types
- data_subset_by_region: Filter by spatial region
- data_subset_by_sample: Filter by sample ID
- data_list_sessions: List active sessions
- data_delete_session: Remove session
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


# --- Pydantic Models for Structured Output ---


class DataInfo(BaseModel):
    """Information about a loaded dataset."""

    session_id: str = Field(description="Session identifier")
    data_key: str = Field(description="Key used to store the data")
    n_cells: int = Field(description="Total number of cells")
    n_dims: int = Field(description="Number of spatial dimensions (2 or 3)")
    cell_types: List[str] = Field(description="List of unique cell type names")
    cell_type_counts: Dict[str, int] = Field(description="Cell counts per type")
    has_markers: bool = Field(description="Whether marker data is available")
    marker_names: Optional[List[str]] = Field(description="Names of marker columns")
    bounds: Dict[str, Dict[str, float]] = Field(
        description="Spatial bounds {dim: {min, max}}"
    )
    has_sample_ids: bool = Field(description="Whether multi-sample data")
    sample_ids: Optional[List[str]] = Field(description="Unique sample IDs if present")


class SessionInfo(BaseModel):
    """Information about a session."""

    session_id: str
    created_at: str
    last_accessed: str
    n_data: int = Field(description="Number of stored datasets")
    n_graphs: int = Field(description="Number of stored graphs")
    n_models: int = Field(description="Number of stored models")
    data_keys: List[str] = Field(description="Keys of stored datasets")


class SessionList(BaseModel):
    """List of sessions."""

    sessions: List[SessionInfo]
    total: int


class DeleteResult(BaseModel):
    """Result of a delete operation."""

    success: bool
    message: str


class CellTypeCounts(BaseModel):
    """Cell type counts."""

    session_id: str
    data_key: str
    counts: Dict[str, int]
    total: int


class SpatialBounds(BaseModel):
    """Spatial bounds of the data."""

    session_id: str
    data_key: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: Optional[float] = None
    z_max: Optional[float] = None


class MarkerInfo(BaseModel):
    """Information about markers."""

    session_id: str
    data_key: str
    has_markers: bool
    marker_names: List[str]
    n_markers: int


# --- Tool Registration ---


def register_tools(mcp: "FastMCP") -> None:
    """Register data tools with the MCP server."""

    @mcp.tool()
    def data_load_csv(
        file_path: str,
        x_col: str = "x",
        y_col: str = "y",
        z_col: Optional[str] = None,
        cell_type_col: str = "cell_type",
        sample_col: Optional[str] = None,
        marker_cols: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        data_key: str = "primary",
    ) -> DataInfo:
        """
        Load spatial tissue data from a CSV file.

        This is typically the first tool to call. It loads cell data including
        spatial coordinates, cell type labels, and optionally marker expression.

        Parameters
        ----------
        file_path : str
            Path to CSV file. Can be absolute or relative to data directory.
        x_col : str
            Column name for X coordinates. Default: "x"
        y_col : str
            Column name for Y coordinates. Default: "y"
        z_col : str, optional
            Column name for Z coordinates (for 3D data).
        cell_type_col : str
            Column name for cell type labels. Default: "cell_type"
        sample_col : str, optional
            Column name for sample IDs (for multi-sample cohort data).
        marker_cols : list of str, optional
            Column names for marker expression. If None, auto-detects.
        session_id : str, optional
            Session to store data in. Creates new session if not provided.
        data_key : str
            Key to store the data under. Default: "primary"

        Returns
        -------
        DataInfo
            Summary of the loaded dataset including cell counts and types.

        Examples
        --------
        Load simple 2D data:
            data_load_csv("/path/to/cells.csv")

        Load with custom columns:
            data_load_csv("/path/to/data.csv", x_col="X_centroid", y_col="Y_centroid")

        Load multi-sample data:
            data_load_csv("/path/to/cohort.csv", sample_col="patient_id")
        """
        from spatialtissuepy.io import read_csv
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        # Resolve file path
        path = resolve_data_path(file_path)

        # Load data
        data = read_csv(
            str(path),
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            celltype_col=cell_type_col,
            sample_col=sample_col,
            marker_cols=marker_cols,
        )

        # Store in session
        session_mgr.store_data(session_id, data_key, data)

        # Build bounds dict
        bounds = {
            "x": {"min": float(data.coordinates[:, 0].min()), "max": float(data.coordinates[:, 0].max())},
            "y": {"min": float(data.coordinates[:, 1].min()), "max": float(data.coordinates[:, 1].max())},
        }
        if data.n_dims == 3:
            bounds["z"] = {"min": float(data.coordinates[:, 2].min()), "max": float(data.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=data_key,
            n_cells=data.n_cells,
            n_dims=data.n_dims,
            cell_types=list(data.cell_types_unique),
            cell_type_counts=dict(data.cell_type_counts),
            has_markers=data.markers is not None,
            marker_names=list(data.markers.columns) if data.markers is not None else None,
            bounds=bounds,
            has_sample_ids=data.sample_ids is not None,
            sample_ids=list(data.sample_ids_unique) if data.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_load_json(
        file_path: str,
        x_key: str = "x",
        y_key: str = "y",
        z_key: Optional[str] = None,
        cell_type_key: str = "cell_type",
        sample_key: Optional[str] = None,
        session_id: Optional[str] = None,
        data_key: str = "primary",
    ) -> DataInfo:
        """
        Load spatial tissue data from a JSON file.

        Expects JSON with a 'cells' array or flat array of cell objects.

        Parameters
        ----------
        file_path : str
            Path to JSON file.
        x_key, y_key : str
            Keys for coordinates in each cell object.
        z_key : str, optional
            Key for Z coordinate (3D data).
        cell_type_key : str
            Key for cell type label.
        sample_key : str, optional
            Key for sample ID.
        session_id : str, optional
            Session to store data in.
        data_key : str
            Key to store the data under.

        Returns
        -------
        DataInfo
            Summary of the loaded dataset.
        """
        from spatialtissuepy.io import read_json
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        path = resolve_data_path(file_path)

        data = read_json(
            str(path),
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
            celltype_key=cell_type_key,
            sample_key=sample_key,
        )

        session_mgr.store_data(session_id, data_key, data)

        bounds = {
            "x": {"min": float(data.coordinates[:, 0].min()), "max": float(data.coordinates[:, 0].max())},
            "y": {"min": float(data.coordinates[:, 1].min()), "max": float(data.coordinates[:, 1].max())},
        }
        if data.n_dims == 3:
            bounds["z"] = {"min": float(data.coordinates[:, 2].min()), "max": float(data.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=data_key,
            n_cells=data.n_cells,
            n_dims=data.n_dims,
            cell_types=list(data.cell_types_unique),
            cell_type_counts=dict(data.cell_type_counts),
            has_markers=data.markers is not None,
            marker_names=list(data.markers.columns) if data.markers is not None else None,
            bounds=bounds,
            has_sample_ids=data.sample_ids is not None,
            sample_ids=list(data.sample_ids_unique) if data.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_save_csv(
        session_id: str,
        output_path: str,
        data_key: str = "primary",
    ) -> Dict[str, Any]:
        """
        Save spatial data to a CSV file.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        output_path : str
            Path for output CSV file.
        data_key : str
            Key of data to export.

        Returns
        -------
        dict
            Confirmation with file path and row count.
        """
        from spatialtissuepy.io import write_csv
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        path = resolve_data_path(output_path)
        write_csv(data, str(path))

        return {
            "success": True,
            "file_path": str(path),
            "n_rows": data.n_cells,
        }

    @mcp.tool()
    def data_save_json(
        session_id: str,
        output_path: str,
        data_key: str = "primary",
    ) -> Dict[str, Any]:
        """
        Save spatial data to a JSON file.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        output_path : str
            Path for output JSON file.
        data_key : str
            Key of data to export.

        Returns
        -------
        dict
            Confirmation with file path and cell count.
        """
        from spatialtissuepy.io import write_json
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        path = resolve_data_path(output_path)
        write_json(data, str(path))

        return {
            "success": True,
            "file_path": str(path),
            "n_cells": data.n_cells,
        }

    @mcp.tool()
    def data_get_info(
        session_id: str,
        data_key: str = "primary",
    ) -> DataInfo:
        """
        Get detailed information about a loaded dataset.

        Use this to understand your data before running analyses.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data to inspect.

        Returns
        -------
        DataInfo
            Comprehensive dataset summary.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        bounds = {
            "x": {"min": float(data.coordinates[:, 0].min()), "max": float(data.coordinates[:, 0].max())},
            "y": {"min": float(data.coordinates[:, 1].min()), "max": float(data.coordinates[:, 1].max())},
        }
        if data.n_dims == 3:
            bounds["z"] = {"min": float(data.coordinates[:, 2].min()), "max": float(data.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=data_key,
            n_cells=data.n_cells,
            n_dims=data.n_dims,
            cell_types=list(data.cell_types_unique),
            cell_type_counts=dict(data.cell_type_counts),
            has_markers=data.markers is not None,
            marker_names=list(data.markers.columns) if data.markers is not None else None,
            bounds=bounds,
            has_sample_ids=data.sample_ids is not None,
            sample_ids=list(data.sample_ids_unique) if data.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_get_cell_types(
        session_id: str,
        data_key: str = "primary",
    ) -> List[str]:
        """
        Get list of unique cell types in the dataset.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.

        Returns
        -------
        list of str
            Unique cell type names.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        return list(data.cell_types_unique)

    @mcp.tool()
    def data_get_cell_counts(
        session_id: str,
        data_key: str = "primary",
    ) -> CellTypeCounts:
        """
        Get cell counts by type.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.

        Returns
        -------
        CellTypeCounts
            Counts per cell type and total.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        counts = dict(data.cell_type_counts)

        return CellTypeCounts(
            session_id=session_id,
            data_key=data_key,
            counts=counts,
            total=data.n_cells,
        )

    @mcp.tool()
    def data_get_bounds(
        session_id: str,
        data_key: str = "primary",
    ) -> SpatialBounds:
        """
        Get spatial bounds of the data.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.

        Returns
        -------
        SpatialBounds
            Min/max coordinates for each dimension.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        result = SpatialBounds(
            session_id=session_id,
            data_key=data_key,
            x_min=float(data.coordinates[:, 0].min()),
            x_max=float(data.coordinates[:, 0].max()),
            y_min=float(data.coordinates[:, 1].min()),
            y_max=float(data.coordinates[:, 1].max()),
        )

        if data.n_dims == 3:
            result.z_min = float(data.coordinates[:, 2].min())
            result.z_max = float(data.coordinates[:, 2].max())

        return result

    @mcp.tool()
    def data_get_markers(
        session_id: str,
        data_key: str = "primary",
    ) -> MarkerInfo:
        """
        Get information about available marker columns.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.

        Returns
        -------
        MarkerInfo
            Marker column names and count.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}' in session '{session_id}'")

        marker_names = list(data.markers.columns) if data.markers is not None else []

        return MarkerInfo(
            session_id=session_id,
            data_key=data_key,
            has_markers=data.markers is not None,
            marker_names=marker_names,
            n_markers=len(marker_names),
        )

    @mcp.tool()
    def data_subset_by_type(
        session_id: str,
        cell_types: List[str],
        data_key: str = "primary",
        output_key: Optional[str] = None,
    ) -> DataInfo:
        """
        Create a subset containing only specified cell types.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        cell_types : list of str
            Cell types to keep.
        data_key : str
            Key of source data.
        output_key : str, optional
            Key for subset data. If None, overwrites source.

        Returns
        -------
        DataInfo
            Information about the subsetted data.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        # Get indices for specified cell types
        mask = data.cell_types.isin(cell_types) if hasattr(data.cell_types, 'isin') else [ct in cell_types for ct in data.cell_types]
        subset = data.subset(cell_types=cell_types)

        out_key = output_key or data_key
        session_mgr.store_data(session_id, out_key, subset)

        bounds = {
            "x": {"min": float(subset.coordinates[:, 0].min()), "max": float(subset.coordinates[:, 0].max())},
            "y": {"min": float(subset.coordinates[:, 1].min()), "max": float(subset.coordinates[:, 1].max())},
        }
        if subset.n_dims == 3:
            bounds["z"] = {"min": float(subset.coordinates[:, 2].min()), "max": float(subset.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=out_key,
            n_cells=subset.n_cells,
            n_dims=subset.n_dims,
            cell_types=list(subset.cell_types_unique),
            cell_type_counts=dict(subset.cell_type_counts),
            has_markers=subset.markers is not None,
            marker_names=list(subset.markers.columns) if subset.markers is not None else None,
            bounds=bounds,
            has_sample_ids=subset.sample_ids is not None,
            sample_ids=list(subset.sample_ids_unique) if subset.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_subset_by_region(
        session_id: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        data_key: str = "primary",
        output_key: Optional[str] = None,
    ) -> DataInfo:
        """
        Create a spatial subset within specified bounds.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        x_min, x_max : float
            X coordinate range.
        y_min, y_max : float
            Y coordinate range.
        z_min, z_max : float, optional
            Z coordinate range (for 3D data).
        data_key : str
            Key of source data.
        output_key : str, optional
            Key for subset data.

        Returns
        -------
        DataInfo
            Information about the subsetted data.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        subset = data.subset_region(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
        )

        out_key = output_key or data_key
        session_mgr.store_data(session_id, out_key, subset)

        bounds = {
            "x": {"min": float(subset.coordinates[:, 0].min()), "max": float(subset.coordinates[:, 0].max())},
            "y": {"min": float(subset.coordinates[:, 1].min()), "max": float(subset.coordinates[:, 1].max())},
        }
        if subset.n_dims == 3:
            bounds["z"] = {"min": float(subset.coordinates[:, 2].min()), "max": float(subset.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=out_key,
            n_cells=subset.n_cells,
            n_dims=subset.n_dims,
            cell_types=list(subset.cell_types_unique),
            cell_type_counts=dict(subset.cell_type_counts),
            has_markers=subset.markers is not None,
            marker_names=list(subset.markers.columns) if subset.markers is not None else None,
            bounds=bounds,
            has_sample_ids=subset.sample_ids is not None,
            sample_ids=list(subset.sample_ids_unique) if subset.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_subset_by_sample(
        session_id: str,
        sample_id: str,
        data_key: str = "primary",
        output_key: Optional[str] = None,
    ) -> DataInfo:
        """
        Extract data for a specific sample from multi-sample data.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        sample_id : str
            Sample ID to extract.
        data_key : str
            Key of source data.
        output_key : str, optional
            Key for subset data.

        Returns
        -------
        DataInfo
            Information about the extracted sample.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if data.sample_ids is None:
            raise ValueError("Data does not contain sample IDs")

        subset = data.subset_sample(sample_id)

        out_key = output_key or f"sample_{sample_id}"
        session_mgr.store_data(session_id, out_key, subset)

        bounds = {
            "x": {"min": float(subset.coordinates[:, 0].min()), "max": float(subset.coordinates[:, 0].max())},
            "y": {"min": float(subset.coordinates[:, 1].min()), "max": float(subset.coordinates[:, 1].max())},
        }
        if subset.n_dims == 3:
            bounds["z"] = {"min": float(subset.coordinates[:, 2].min()), "max": float(subset.coordinates[:, 2].max())}

        return DataInfo(
            session_id=session_id,
            data_key=out_key,
            n_cells=subset.n_cells,
            n_dims=subset.n_dims,
            cell_types=list(subset.cell_types_unique),
            cell_type_counts=dict(subset.cell_type_counts),
            has_markers=subset.markers is not None,
            marker_names=list(subset.markers.columns) if subset.markers is not None else None,
            bounds=bounds,
            has_sample_ids=subset.sample_ids is not None,
            sample_ids=list(subset.sample_ids_unique) if subset.sample_ids is not None else None,
        )

    @mcp.tool()
    def data_list_sessions() -> SessionList:
        """
        List all active sessions with their metadata.

        Returns
        -------
        SessionList
            Information about all sessions.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        sessions = session_mgr.list_sessions()

        session_infos = []
        for s in sessions:
            session_infos.append(SessionInfo(
                session_id=s["session_id"],
                created_at=s["created_at"],
                last_accessed=s["last_accessed"],
                n_data=s["n_data"],
                n_graphs=s["n_graphs"],
                n_models=s["n_models"],
                data_keys=session_mgr.list_data(s["session_id"]),
            ))

        return SessionList(
            sessions=session_infos,
            total=len(session_infos),
        )

    @mcp.tool()
    def data_delete_session(
        session_id: str,
    ) -> DeleteResult:
        """
        Delete a session and all its stored data.

        Parameters
        ----------
        session_id : str
            Session to delete.

        Returns
        -------
        DeleteResult
            Success status and message.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        success = session_mgr.delete_session(session_id)

        if success:
            return DeleteResult(
                success=True,
                message=f"Session '{session_id}' deleted successfully",
            )
        else:
            return DeleteResult(
                success=False,
                message=f"Session '{session_id}' not found",
            )
