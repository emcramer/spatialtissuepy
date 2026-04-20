"""
Spatial tools for MCP server.

Tools for distance calculations and neighborhood operations.

Tools (7 total):
- spatial_pairwise_distances: Full distance matrix
- spatial_nearest_neighbors: k-NN indices and distances
- spatial_radius_neighbors: Neighbors within radius
- spatial_density: Local cell density
- spatial_boundary_cells: Identify boundary cells
- spatial_convex_hull: Compute convex hull
- spatial_voronoi_areas: Voronoi cell areas
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


# --- Pydantic Models ---


class DistanceMatrixResult(BaseModel):
    """Result of distance matrix computation."""

    session_id: str
    data_key: str
    shape: List[int] = Field(description="Matrix shape [n_cells, n_cells]")
    min_distance: float
    max_distance: float
    mean_distance: float
    stored_key: Optional[str] = Field(description="Cache key if stored")


class NearestNeighborsResult(BaseModel):
    """Result of k-nearest neighbors query."""

    session_id: str
    data_key: str
    k: int
    n_cells: int
    mean_nn_distance: float = Field(description="Mean distance to nearest neighbor")
    median_nn_distance: float
    max_nn_distance: float
    distances_summary: Dict[str, float] = Field(
        description="Summary statistics of NN distances"
    )


class RadiusNeighborsResult(BaseModel):
    """Result of radius neighbors query."""

    session_id: str
    data_key: str
    radius: float
    n_cells: int
    mean_neighbors: float = Field(description="Mean number of neighbors per cell")
    median_neighbors: float
    max_neighbors: int
    min_neighbors: int
    cells_with_no_neighbors: int


class DensityResult(BaseModel):
    """Result of density computation."""

    session_id: str
    data_key: str
    radius: float
    n_cells: int
    min_density: float
    max_density: float
    mean_density: float
    std_density: float


class BoundaryCellsResult(BaseModel):
    """Result of boundary cell detection."""

    session_id: str
    data_key: str
    n_boundary_cells: int
    n_total_cells: int
    boundary_fraction: float
    method: str


class ConvexHullResult(BaseModel):
    """Result of convex hull computation."""

    session_id: str
    data_key: str
    area: float
    perimeter: float
    n_vertices: int
    vertices: List[List[float]] = Field(description="Hull vertex coordinates")


class VoronoiResult(BaseModel):
    """Result of Voronoi area computation."""

    session_id: str
    data_key: str
    n_cells: int
    mean_area: float
    median_area: float
    min_area: float
    max_area: float
    total_area: float


# --- Tool Registration ---


def register_tools(mcp: "FastMCP") -> None:
    """Register spatial tools with the MCP server."""

    @mcp.tool()
    def spatial_pairwise_distances(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        max_cells: int = 5000,
    ) -> DistanceMatrixResult:
        """
        Compute pairwise distance matrix between cells.

        For large datasets, consider using spatial_nearest_neighbors instead.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute only for this cell type.
        max_cells : int
            Maximum cells to compute (for memory safety).

        Returns
        -------
        DistanceMatrixResult
            Summary statistics of the distance matrix.
        """
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        # Optionally subset by cell type
        if cell_type:
            data = data.subset(cell_types=[cell_type])

        if data.n_cells > max_cells:
            raise ValueError(
                f"Dataset has {data.n_cells} cells, exceeding max_cells={max_cells}. "
                "Use spatial_nearest_neighbors for large datasets."
            )

        # Compute distances
        distances = pdist(data.coordinates)
        dist_matrix = squareform(distances)

        # Get non-diagonal values for statistics
        mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
        non_diag = dist_matrix[mask]

        return DistanceMatrixResult(
            session_id=session_id,
            data_key=data_key,
            shape=list(dist_matrix.shape),
            min_distance=float(non_diag.min()),
            max_distance=float(non_diag.max()),
            mean_distance=float(non_diag.mean()),
            stored_key=None,
        )

    @mcp.tool()
    def spatial_nearest_neighbors(
        session_id: str,
        k: int = 10,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
    ) -> NearestNeighborsResult:
        """
        Find k nearest neighbors for each cell.

        Uses KD-tree for efficient computation on large datasets.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        k : int
            Number of neighbors to find.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute only for this cell type.

        Returns
        -------
        NearestNeighborsResult
            Summary of nearest neighbor distances.
        """
        import numpy as np
        from spatialtissuepy.spatial import nearest_neighbors
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        # Compute nearest neighbors
        distances, indices = nearest_neighbors(data.coordinates, k=k)

        # Distance to first nearest neighbor (excluding self)
        nn1_distances = distances[:, 0] if distances.shape[1] > 0 else distances.flatten()

        return NearestNeighborsResult(
            session_id=session_id,
            data_key=data_key,
            k=k,
            n_cells=data.n_cells,
            mean_nn_distance=float(np.mean(nn1_distances)),
            median_nn_distance=float(np.median(nn1_distances)),
            max_nn_distance=float(np.max(nn1_distances)),
            distances_summary={
                "mean": float(np.mean(distances)),
                "std": float(np.std(distances)),
                "min": float(np.min(distances)),
                "max": float(np.max(distances)),
                "percentile_25": float(np.percentile(distances, 25)),
                "percentile_75": float(np.percentile(distances, 75)),
            },
        )

    @mcp.tool()
    def spatial_radius_neighbors(
        session_id: str,
        radius: float,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
    ) -> RadiusNeighborsResult:
        """
        Find all neighbors within a radius for each cell.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        radius : float
            Search radius (in coordinate units).
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute only for this cell type.

        Returns
        -------
        RadiusNeighborsResult
            Summary of neighbor counts per cell.
        """
        import numpy as np
        from spatialtissuepy.spatial import radius_neighbors
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        # Compute radius neighbors
        neighbors = radius_neighbors(data.coordinates, radius=radius)

        # Count neighbors per cell (excluding self)
        neighbor_counts = np.array([len(n) for n in neighbors])

        return RadiusNeighborsResult(
            session_id=session_id,
            data_key=data_key,
            radius=radius,
            n_cells=data.n_cells,
            mean_neighbors=float(np.mean(neighbor_counts)),
            median_neighbors=float(np.median(neighbor_counts)),
            max_neighbors=int(np.max(neighbor_counts)),
            min_neighbors=int(np.min(neighbor_counts)),
            cells_with_no_neighbors=int(np.sum(neighbor_counts == 0)),
        )

    @mcp.tool()
    def spatial_density(
        session_id: str,
        radius: float = 50.0,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
    ) -> DensityResult:
        """
        Compute local cell density for each cell.

        Density is the number of neighbors within the radius.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        radius : float
            Radius for density calculation.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute density only for this cell type.

        Returns
        -------
        DensityResult
            Summary statistics of cell densities.
        """
        import numpy as np
        from spatialtissuepy.spatial import radius_neighbors
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        # Compute local density as neighbor counts within radius
        neighbors = radius_neighbors(data.coordinates, radius=radius)
        density = np.array([len(n) for n in neighbors])

        return DensityResult(
            session_id=session_id,
            data_key=data_key,
            radius=radius,
            n_cells=data.n_cells,
            min_density=float(np.min(density)),
            max_density=float(np.max(density)),
            mean_density=float(np.mean(density)),
            std_density=float(np.std(density)),
        )

    @mcp.tool()
    def spatial_boundary_cells(
        session_id: str,
        data_key: str = "primary",
        method: str = "alpha_shape",
        alpha: float = 0.0,
    ) -> BoundaryCellsResult:
        """
        Identify cells on the boundary of the tissue.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        method : str
            Detection method: "alpha_shape" or "convex_hull".
        alpha : float
            Alpha parameter for alpha shape (0 = convex hull).

        Returns
        -------
        BoundaryCellsResult
            Information about boundary cells.
        """
        import numpy as np
        from spatialtissuepy.spatial import boundary_cells
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        # Detect boundary cells
        boundary_mask = boundary_cells(data, method=method, alpha=alpha)
        n_boundary = int(np.sum(boundary_mask))

        return BoundaryCellsResult(
            session_id=session_id,
            data_key=data_key,
            n_boundary_cells=n_boundary,
            n_total_cells=data.n_cells,
            boundary_fraction=n_boundary / data.n_cells,
            method=method,
        )

    @mcp.tool()
    def spatial_convex_hull(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
    ) -> ConvexHullResult:
        """
        Compute the convex hull of cell positions.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute hull only for this cell type.

        Returns
        -------
        ConvexHullResult
            Hull area, perimeter, and vertices.
        """
        import numpy as np
        from scipy.spatial import ConvexHull
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        if data.n_cells < 3:
            raise ValueError("Need at least 3 cells to compute convex hull")

        # Compute convex hull (2D)
        coords_2d = data.coordinates[:, :2]
        hull = ConvexHull(coords_2d)

        vertices = coords_2d[hull.vertices].tolist()

        return ConvexHullResult(
            session_id=session_id,
            data_key=data_key,
            area=float(hull.volume),  # In 2D, 'volume' is area
            perimeter=float(hull.area),  # In 2D, 'area' is perimeter
            n_vertices=len(hull.vertices),
            vertices=vertices,
        )

    @mcp.tool()
    def spatial_voronoi_areas(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
    ) -> VoronoiResult:
        """
        Compute Voronoi cell areas for each cell.

        Voronoi area represents the "territory" of each cell.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Compute only for this cell type.

        Returns
        -------
        VoronoiResult
            Summary statistics of Voronoi areas.
        """
        import numpy as np
        from scipy.spatial import Voronoi
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        if data.n_cells < 4:
            raise ValueError("Need at least 4 cells for Voronoi tessellation")

        # Compute Voronoi tessellation (2D)
        coords_2d = data.coordinates[:, :2]
        vor = Voronoi(coords_2d)

        # Compute areas for finite regions
        areas = []
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                # Finite region - compute area
                polygon = vor.vertices[region]
                # Shoelace formula
                n = len(polygon)
                area = 0.5 * abs(sum(
                    polygon[i][0] * polygon[(i + 1) % n][1] -
                    polygon[(i + 1) % n][0] * polygon[i][1]
                    for i in range(n)
                ))
                areas.append(area)
            else:
                # Infinite region - use NaN or estimate
                areas.append(np.nan)

        areas = np.array(areas)
        finite_areas = areas[~np.isnan(areas)]

        if len(finite_areas) == 0:
            raise ValueError("No finite Voronoi regions found")

        return VoronoiResult(
            session_id=session_id,
            data_key=data_key,
            n_cells=data.n_cells,
            mean_area=float(np.mean(finite_areas)),
            median_area=float(np.median(finite_areas)),
            min_area=float(np.min(finite_areas)),
            max_area=float(np.max(finite_areas)),
            total_area=float(np.sum(finite_areas)),
        )
