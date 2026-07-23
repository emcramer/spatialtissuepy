"""
Boundary and interface detection.

A boundary cell sits at the interface between regions of different identity --
different cell types, or different niches. Each cell's *foreignness* is the
fraction of its spatial neighbors whose label differs from its own; cells above
a threshold are boundary cells. Passing niche labels (from
:func:`~spatialtissuepy.microenvironment.niches.identify_niches`) detects niche
interfaces; the default uses cell type, generalizing the pairwise
:func:`~spatialtissuepy.spatial.interface_cells` to any grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import SpatialTissueData
from ..spatial import compute_neighborhoods


@dataclass
class BoundaryResult:
    """
    Result of :func:`detect_boundaries`.

    Attributes
    ----------
    is_boundary : np.ndarray
        ``(n_cells,)`` boolean mask; ``True`` for boundary cells.
    foreignness : np.ndarray
        ``(n_cells,)`` fraction of each cell's neighbors whose label differs
        from its own. ``0`` for cells with too few neighbors.
    labels : np.ndarray
        ``(n_cells,)`` grouping labels used for the comparison.
    radius : float
        Neighborhood radius used.
    """

    is_boundary: np.ndarray
    foreignness: np.ndarray
    labels: np.ndarray
    radius: float

    @property
    def boundary_indices(self) -> np.ndarray:
        """Indices of the boundary cells."""
        return np.where(self.is_boundary)[0]

    def boundary_fraction(self) -> float:
        """Fraction of all cells that are boundary cells."""
        if self.is_boundary.size == 0:
            return 0.0
        return float(np.mean(self.is_boundary))


def detect_boundaries(
    data: SpatialTissueData,
    radius: float,
    labels: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    min_neighbors: int = 1,
) -> BoundaryResult:
    """
    Detect cells at the interface between groups.

    For each cell, ``foreignness`` is the fraction of neighbors within
    ``radius`` whose group label differs from the cell's own. A cell is a
    boundary cell when its foreignness exceeds ``threshold`` and it has at least
    ``min_neighbors`` neighbors.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Neighborhood radius.
    labels : np.ndarray, optional
        ``(n_cells,)`` group label per cell. Defaults to the cell types. Pass
        niche labels to detect niche interfaces.
    threshold : float, default 0.0
        Minimum foreignness (exclusive) to call a cell a boundary. The default
        of ``0.0`` flags any cell with at least one differing neighbor; raise it
        to require a more mixed neighborhood.
    min_neighbors : int, default 1
        Minimum neighbor count; cells with fewer are never boundary cells and
        get foreignness ``0``.

    Returns
    -------
    BoundaryResult

    Raises
    ------
    ValueError
        If ``labels`` is given but its length does not match ``n_cells``.
    """
    if labels is None:
        labels = np.asarray(data.cell_types)
    else:
        labels = np.asarray(labels)
        if labels.shape[0] != data.n_cells:
            raise ValueError(
                f"labels length {labels.shape[0]} != n_cells {data.n_cells}"
            )

    neighborhoods = compute_neighborhoods(
        data, method='radius', radius=radius, include_self=False
    )

    n = data.n_cells
    foreignness = np.zeros(n)
    is_boundary = np.zeros(n, dtype=bool)

    for i, neighbors in enumerate(neighborhoods):
        neighbors = np.asarray(neighbors)
        if neighbors.size < min_neighbors:
            continue
        differing = np.count_nonzero(labels[neighbors] != labels[i])
        frac = differing / neighbors.size
        foreignness[i] = frac
        if frac > threshold:
            is_boundary[i] = True

    return BoundaryResult(
        is_boundary=is_boundary,
        foreignness=foreignness,
        labels=labels,
        radius=radius,
    )
