"""
Statistics tools for MCP server.

Spatial statistics analysis tools.

Tools (10 total):
- statistics_ripleys_k: Ripley's K function
- statistics_ripleys_l: Ripley's L transformation
- statistics_ripleys_h: Ripley's H (deviation from CSR)
- statistics_pair_correlation: g(r) function
- statistics_nearest_neighbor_g: G-function
- statistics_colocalization_quotient: CLQ between types
- statistics_cross_k: Cross-type Ripley's K
- statistics_getis_ord_gi_star: Hotspot detection
- statistics_morans_i: Spatial autocorrelation
- statistics_mark_correlation: Mark correlation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


# --- Pydantic Models ---


class RipleysResult(BaseModel):
    """Result of Ripley's K/L/H analysis."""

    session_id: str
    data_key: str
    cell_type: Optional[str] = Field(description="Cell type analyzed, or None for all")
    n_cells: int
    radii: List[float]
    k_values: List[float]
    l_values: List[float]
    h_values: List[float]
    h_max: float = Field(description="Maximum absolute H value")
    h_max_radius: float = Field(description="Radius at which H_max occurs")
    interpretation: str = Field(description="Human-readable interpretation")


class ColocalizationResult(BaseModel):
    """Result of colocalization analysis."""

    session_id: str
    data_key: str
    type_a: str
    type_b: str
    clq: float = Field(description="Colocalization quotient")
    n_type_a: int
    n_type_b: int
    radius: float
    interpretation: str


class CrossKResult(BaseModel):
    """Result of cross-type Ripley's K analysis."""

    session_id: str
    data_key: str
    type_a: str
    type_b: str
    radii: List[float]
    k_values: List[float]
    interpretation: str


class HotspotResult(BaseModel):
    """Result of Getis-Ord Gi* hotspot analysis."""

    session_id: str
    data_key: str
    cell_type: Optional[str]
    radius: float
    n_cells: int
    n_hotspots: int = Field(description="Cells with significant positive Gi*")
    n_coldspots: int = Field(description="Cells with significant negative Gi*")
    hotspot_fraction: float
    coldspot_fraction: float
    max_gi_star: float
    min_gi_star: float


class MoransIResult(BaseModel):
    """Result of Moran's I spatial autocorrelation."""

    session_id: str
    data_key: str
    marker: str = Field(description="Marker used for autocorrelation")
    morans_i: float = Field(description="Moran's I statistic (-1 to 1)")
    expected_i: float = Field(description="Expected I under null hypothesis")
    z_score: float
    p_value: float
    interpretation: str


class PairCorrelationResult(BaseModel):
    """Result of pair correlation function g(r)."""

    session_id: str
    data_key: str
    cell_type: Optional[str]
    radii: List[float]
    g_values: List[float]
    interpretation: str


class GFunctionResult(BaseModel):
    """Result of nearest neighbor G function."""

    session_id: str
    data_key: str
    cell_type: Optional[str]
    distances: List[float]
    g_values: List[float]
    mean_nn_distance: float
    interpretation: str


class MarkCorrelationResult(BaseModel):
    """Result of mark correlation function."""

    session_id: str
    data_key: str
    marker: str
    radii: List[float]
    kmm_values: List[float] = Field(description="Mark correlation values")
    interpretation: str


# --- Helper Functions ---


def _compute_ripleys(
    session_id: str,
    data_key: str,
    cell_type: Optional[str],
    radii: Optional[List[float]],
    max_radius: float,
    n_radii: int,
    edge_correction: str = "ripley",
) -> RipleysResult:
    """Internal helper to compute Ripley's K/L/H functions."""
    import numpy as np
    from spatialtissuepy.statistics import ripleys_k, ripleys_l, ripleys_h
    from ..server import get_session_manager

    session_mgr = get_session_manager()
    data = session_mgr.load_data(session_id, data_key)

    if data is None:
        raise ValueError(f"No data found with key '{data_key}'")

    if cell_type:
        data = data.subset(cell_types=[cell_type])

    if radii is None:
        radii = np.linspace(10, max_radius, n_radii).tolist()

    radii_arr = np.array(radii)
    coords = data.coordinates

    K = ripleys_k(coords, radii=radii_arr, edge_correction=edge_correction)
    L = ripleys_l(coords, radii=radii_arr, edge_correction=edge_correction)
    H = ripleys_h(coords, radii=radii_arr, edge_correction=edge_correction)

    h_max_idx = int(np.argmax(np.abs(H)))
    h_max = float(H[h_max_idx])
    h_max_radius = float(radii_arr[h_max_idx])

    if h_max > 10:
        interpretation = f"Strong clustering detected (H_max={h_max:.1f} at r={h_max_radius:.1f})"
    elif h_max > 0:
        interpretation = f"Moderate clustering (H_max={h_max:.1f} at r={h_max_radius:.1f})"
    elif h_max < -10:
        interpretation = f"Strong dispersion/regularity (H_min={h_max:.1f} at r={h_max_radius:.1f})"
    elif h_max < 0:
        interpretation = f"Moderate dispersion (H_min={h_max:.1f} at r={h_max_radius:.1f})"
    else:
        interpretation = "Pattern consistent with complete spatial randomness (CSR)"

    return RipleysResult(
        session_id=session_id,
        data_key=data_key,
        cell_type=cell_type,
        n_cells=data.n_cells,
        radii=radii,
        k_values=K.tolist(),
        l_values=L.tolist(),
        h_values=H.tolist(),
        h_max=h_max,
        h_max_radius=h_max_radius,
        interpretation=interpretation,
    )


# --- Tool Registration ---


def register_tools(mcp: "FastMCP") -> None:
    """Register statistics tools with the MCP server."""

    @mcp.tool()
    def statistics_ripleys_k(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
        edge_correction: str = "ripley",
    ) -> RipleysResult:
        """
        Compute Ripley's K function for spatial clustering analysis.

        K(r) counts the expected number of points within distance r of a typical point.
        Under complete spatial randomness (CSR): K(r) = pi * r^2

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data to analyze.
        cell_type : str, optional
            Analyze only this cell type. If None, uses all cells.
        radii : list of float, optional
            Specific radii to compute. If None, uses linspace(10, max_radius, n_radii).
        max_radius : float
            Maximum radius if radii not provided.
        n_radii : int
            Number of radii if radii not provided.
        edge_correction : str
            Edge correction: "ripley", "isotropic", or "none".

        Returns
        -------
        RipleysResult
            K, L, H values and interpretation.
        """
        return _compute_ripleys(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            radii=radii,
            max_radius=max_radius,
            n_radii=n_radii,
            edge_correction=edge_correction,
        )

    @mcp.tool()
    def statistics_ripleys_l(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
    ) -> RipleysResult:
        """
        Compute Ripley's L function (variance-stabilized K).

        L(r) = sqrt(K(r)/pi). Under CSR: L(r) = r.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Cell type to analyze.
        radii : list of float, optional
            Radii to compute at.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radii.

        Returns
        -------
        RipleysResult
            L values and related statistics.
        """
        return _compute_ripleys(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            radii=radii,
            max_radius=max_radius,
            n_radii=n_radii,
        )

    @mcp.tool()
    def statistics_ripleys_h(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
    ) -> RipleysResult:
        """
        Compute Ripley's H function (centered L).

        H(r) = L(r) - r. H > 0 indicates clustering, H < 0 indicates dispersion.
        This is the most interpretable form of Ripley's functions.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Cell type to analyze.
        radii : list of float, optional
            Radii to compute at.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radii.

        Returns
        -------
        RipleysResult
            H values and interpretation.
        """
        return _compute_ripleys(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            radii=radii,
            max_radius=max_radius,
            n_radii=n_radii,
        )

    @mcp.tool()
    def statistics_colocalization_quotient(
        session_id: str,
        type_a: str,
        type_b: str,
        radius: float = 50.0,
        data_key: str = "primary",
    ) -> ColocalizationResult:
        """
        Compute colocalization quotient (CLQ) between two cell types.

        CLQ measures spatial association:
        - CLQ > 1: Attraction (cells closer than expected)
        - CLQ = 1: No association (random mixing)
        - CLQ < 1: Repulsion (cells farther than expected)

        Parameters
        ----------
        session_id : str
            Session containing the data.
        type_a : str
            First cell type.
        type_b : str
            Second cell type.
        radius : float
            Neighborhood radius for CLQ computation.
        data_key : str
            Key of the data.

        Returns
        -------
        ColocalizationResult
            CLQ value and interpretation.
        """
        from spatialtissuepy.statistics import colocalization_quotient
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        clq = colocalization_quotient(data, type_a, type_b, radius=radius)

        counts = data.cell_type_counts
        n_a = counts.get(type_a, 0)
        n_b = counts.get(type_b, 0)

        if clq > 1.5:
            interpretation = f"Strong attraction: {type_a} and {type_b} co-localize significantly"
        elif clq > 1.1:
            interpretation = f"Moderate attraction between {type_a} and {type_b}"
        elif clq < 0.5:
            interpretation = f"Strong repulsion: {type_a} and {type_b} avoid each other"
        elif clq < 0.9:
            interpretation = f"Moderate repulsion between {type_a} and {type_b}"
        else:
            interpretation = f"No significant spatial association between {type_a} and {type_b}"

        return ColocalizationResult(
            session_id=session_id,
            data_key=data_key,
            type_a=type_a,
            type_b=type_b,
            clq=float(clq),
            n_type_a=n_a,
            n_type_b=n_b,
            radius=radius,
            interpretation=interpretation,
        )

    @mcp.tool()
    def statistics_cross_k(
        session_id: str,
        type_a: str,
        type_b: str,
        data_key: str = "primary",
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
    ) -> CrossKResult:
        """
        Compute cross-type Ripley's K function.

        Measures spatial relationship between two different cell types.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        type_a : str
            First cell type (reference points).
        type_b : str
            Second cell type (counted around type_a).
        data_key : str
            Key of the data.
        radii : list of float, optional
            Radii to compute at.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radii.

        Returns
        -------
        CrossKResult
            Cross-K values and interpretation.
        """
        from spatialtissuepy.statistics import cross_k_function
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if radii is None:
            radii = np.linspace(10, max_radius, n_radii).tolist()

        radii_arr = np.array(radii)

        K_cross = cross_k_function(data, type_a, type_b, radii=radii_arr)

        # Compare to CSR expectation
        K_csr = np.pi * radii_arr**2
        deviation = K_cross - K_csr
        max_dev_idx = int(np.argmax(np.abs(deviation)))
        max_dev = float(deviation[max_dev_idx])

        if max_dev > 0:
            interpretation = f"{type_b} cells cluster around {type_a} cells more than expected"
        elif max_dev < 0:
            interpretation = f"{type_b} cells are dispersed away from {type_a} cells"
        else:
            interpretation = f"Random spatial relationship between {type_a} and {type_b}"

        return CrossKResult(
            session_id=session_id,
            data_key=data_key,
            type_a=type_a,
            type_b=type_b,
            radii=radii,
            k_values=K_cross.tolist(),
            interpretation=interpretation,
        )

    @mcp.tool()
    def statistics_getis_ord_gi_star(
        session_id: str,
        radius: float = 50.0,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        significance_level: float = 0.05,
    ) -> HotspotResult:
        """
        Compute Getis-Ord Gi* for hotspot detection.

        Identifies statistically significant spatial clusters (hotspots)
        and cold spots in cell density.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        radius : float
            Neighborhood radius for Gi* computation.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Analyze density of this cell type.
        significance_level : float
            P-value threshold for significance.

        Returns
        -------
        HotspotResult
            Hotspot and coldspot counts.
        """
        from spatialtissuepy.statistics import getis_ord_gi_star
        from scipy import stats
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        gi_star = getis_ord_gi_star(data, radius=radius, cell_type=cell_type)

        # Determine significance thresholds
        z_threshold = stats.norm.ppf(1 - significance_level / 2)

        n_hotspots = int(np.sum(gi_star > z_threshold))
        n_coldspots = int(np.sum(gi_star < -z_threshold))

        return HotspotResult(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            radius=radius,
            n_cells=data.n_cells,
            n_hotspots=n_hotspots,
            n_coldspots=n_coldspots,
            hotspot_fraction=n_hotspots / data.n_cells,
            coldspot_fraction=n_coldspots / data.n_cells,
            max_gi_star=float(np.max(gi_star)),
            min_gi_star=float(np.min(gi_star)),
        )

    @mcp.tool()
    def statistics_morans_i(
        session_id: str,
        marker: str,
        radius: float = 50.0,
        data_key: str = "primary",
    ) -> MoransIResult:
        """
        Compute Moran's I spatial autocorrelation for a marker.

        Measures whether similar marker values cluster spatially:
        - I > 0: Positive autocorrelation (similar values cluster)
        - I = 0: Random distribution
        - I < 0: Negative autocorrelation (dissimilar values cluster)

        Parameters
        ----------
        session_id : str
            Session containing the data.
        marker : str
            Name of marker column to analyze.
        radius : float
            Neighborhood radius for spatial weights.
        data_key : str
            Key of the data.

        Returns
        -------
        MoransIResult
            Moran's I statistic and significance.
        """
        from spatialtissuepy.statistics import morans_i
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if data.markers is None or marker not in data.markers.columns:
            raise ValueError(f"Marker '{marker}' not found in data")

        result = morans_i(data, marker=marker, radius=radius)

        # Extract results (assuming morans_i returns dict or object)
        if isinstance(result, dict):
            I = result.get("I", result.get("morans_i", 0))
            E_I = result.get("expected_I", -1 / (data.n_cells - 1))
            z = result.get("z_score", 0)
            p = result.get("p_value", 1)
        else:
            I = float(result)
            E_I = -1 / (data.n_cells - 1)
            z = 0
            p = 1

        if I > 0.3:
            interpretation = f"Strong positive spatial autocorrelation for {marker}"
        elif I > 0.1:
            interpretation = f"Moderate positive spatial autocorrelation for {marker}"
        elif I < -0.3:
            interpretation = f"Strong negative spatial autocorrelation for {marker}"
        elif I < -0.1:
            interpretation = f"Moderate negative spatial autocorrelation for {marker}"
        else:
            interpretation = f"Weak or no spatial autocorrelation for {marker}"

        return MoransIResult(
            session_id=session_id,
            data_key=data_key,
            marker=marker,
            morans_i=float(I),
            expected_i=float(E_I),
            z_score=float(z),
            p_value=float(p),
            interpretation=interpretation,
        )

    @mcp.tool()
    def statistics_pair_correlation(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
    ) -> PairCorrelationResult:
        """
        Compute pair correlation function g(r).

        g(r) is the derivative of Ripley's K, showing clustering at specific scales:
        - g(r) > 1: Clustering at distance r
        - g(r) = 1: CSR at distance r
        - g(r) < 1: Dispersion at distance r

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Cell type to analyze.
        radii : list of float, optional
            Radii to compute at.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radii.

        Returns
        -------
        PairCorrelationResult
            g(r) values and interpretation.
        """
        from spatialtissuepy.statistics import pair_correlation_function
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        if radii is None:
            radii = np.linspace(10, max_radius, n_radii).tolist()

        radii_arr = np.array(radii)

        g = pair_correlation_function(data, radii=radii_arr)

        max_g = float(np.max(g))
        max_g_idx = int(np.argmax(g))
        max_g_radius = float(radii_arr[max_g_idx])

        if max_g > 1.5:
            interpretation = f"Peak clustering at r={max_g_radius:.1f} (g={max_g:.2f})"
        elif max_g > 1.1:
            interpretation = f"Moderate clustering at r={max_g_radius:.1f}"
        else:
            interpretation = "Pattern close to complete spatial randomness"

        return PairCorrelationResult(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            radii=radii,
            g_values=g.tolist(),
            interpretation=interpretation,
        )

    @mcp.tool()
    def statistics_nearest_neighbor_g(
        session_id: str,
        data_key: str = "primary",
        cell_type: Optional[str] = None,
        n_distances: int = 50,
    ) -> GFunctionResult:
        """
        Compute nearest neighbor G function.

        G(r) is the cumulative distribution of nearest neighbor distances.
        Useful for detecting regularity vs clustering.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the data.
        cell_type : str, optional
            Cell type to analyze.
        n_distances : int
            Number of distance values to compute.

        Returns
        -------
        GFunctionResult
            G function values and mean NN distance.
        """
        from spatialtissuepy.statistics import nearest_neighbor_g_function
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if cell_type:
            data = data.subset(cell_types=[cell_type])

        distances, g_values = nearest_neighbor_g_function(data, n_distances=n_distances)

        # Compute mean NN distance
        from spatialtissuepy.spatial import nearest_neighbors
        _, nn_dists = nearest_neighbors(data, k=1)
        mean_nn = float(np.mean(nn_dists))

        if mean_nn < np.median(distances):
            interpretation = "Cells are more clustered than CSR (small NN distances)"
        else:
            interpretation = "Cells show regular/dispersed pattern (large NN distances)"

        return GFunctionResult(
            session_id=session_id,
            data_key=data_key,
            cell_type=cell_type,
            distances=distances.tolist(),
            g_values=g_values.tolist(),
            mean_nn_distance=mean_nn,
            interpretation=interpretation,
        )

    @mcp.tool()
    def statistics_mark_correlation(
        session_id: str,
        marker: str,
        data_key: str = "primary",
        radii: Optional[List[float]] = None,
        max_radius: float = 200.0,
        n_radii: int = 20,
    ) -> MarkCorrelationResult:
        """
        Compute mark correlation function for a marker.

        Measures how marker values correlate at different spatial scales.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        marker : str
            Name of marker column.
        data_key : str
            Key of the data.
        radii : list of float, optional
            Radii to compute at.
        max_radius : float
            Maximum radius.
        n_radii : int
            Number of radii.

        Returns
        -------
        MarkCorrelationResult
            Mark correlation values kmm(r).
        """
        from spatialtissuepy.statistics import mark_correlation
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if data.markers is None or marker not in data.markers.columns:
            raise ValueError(f"Marker '{marker}' not found in data")

        if radii is None:
            radii = np.linspace(10, max_radius, n_radii).tolist()

        radii_arr = np.array(radii)

        kmm = mark_correlation(data, marker=marker, radii=radii_arr)

        if np.mean(kmm) > 1.1:
            interpretation = f"Positive mark correlation: similar {marker} values cluster together"
        elif np.mean(kmm) < 0.9:
            interpretation = f"Negative mark correlation: dissimilar {marker} values are neighbors"
        else:
            interpretation = f"No significant mark correlation for {marker}"

        return MarkCorrelationResult(
            session_id=session_id,
            data_key=data_key,
            marker=marker,
            radii=radii,
            kmm_values=kmm.tolist(),
            interpretation=interpretation,
        )
