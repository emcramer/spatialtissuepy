"""
Spatial gradient analysis.

Estimates the spatial gradient of a scalar field -- a substrate concentration,
or local cell density -- by fitting a local linear model to the values at each
point's nearest neighbors. The gradient vector points in the direction of
steepest increase, and its magnitude is the rate of change per unit distance.
This is grid-agnostic: it works on the regular voxel mesh of a microenvironment
field and on scattered cell positions alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core import SpatialTissueData
from ..spatial import compute_neighborhoods

if TYPE_CHECKING:
    from ..synthetic.physicell import PhysiCellTimeStep


@dataclass
class GradientField:
    """
    Result of a gradient estimation.

    Attributes
    ----------
    points : np.ndarray
        ``(n_points, n_dims)`` coordinates where the gradient was evaluated.
    gradients : np.ndarray
        ``(n_points, n_dims)`` gradient vector at each point.
    values : np.ndarray
        ``(n_points,)`` field value sampled at each point.
    """

    points: np.ndarray
    gradients: np.ndarray
    values: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        """``(n_points,)`` gradient magnitude (steepness)."""
        return np.linalg.norm(self.gradients, axis=1)

    @property
    def direction(self) -> np.ndarray:
        """
        ``(n_points, n_dims)`` unit vectors along each gradient.

        Points whose gradient magnitude is negligible relative to the steepest
        gradient in the field -- flat regions, and the round-off noise a local
        fit produces on a constant field -- yield zero vectors rather than
        normalized noise or NaNs.
        """
        mag = self.magnitude
        scale = mag.max() if mag.size else 0.0
        if scale <= 1e-12:
            # The whole field is flat; no meaningful directions.
            return np.zeros_like(self.gradients)
        tol = scale * 1e-6
        safe = np.where(mag > tol, mag, 1.0)
        unit = self.gradients / safe[:, None]
        unit[mag <= tol] = 0.0
        return unit


def spatial_gradient(
    positions: np.ndarray,
    values: np.ndarray,
    query_points: Optional[np.ndarray] = None,
    k: int = 12,
) -> GradientField:
    """
    Estimate the gradient of a scalar field by local linear regression.

    Around each query point, the ``k`` nearest source points are found and a
    linear model ``value ~ a + g . (x - q)`` is fit by least squares; the slope
    ``g`` is the gradient estimate. Points are centered on the query for
    numerical stability.

    Parameters
    ----------
    positions : np.ndarray
        ``(n, n_dims)`` coordinates where the field is known.
    values : np.ndarray
        ``(n,)`` field values at ``positions``.
    query_points : np.ndarray, optional
        ``(m, n_dims)`` points at which to evaluate the gradient. Defaults to
        ``positions``. The reported ``values`` are then nearest-source samples.
    k : int, default 12
        Number of neighbors for each local fit. Clamped to the number of source
        points; a fit needs at least ``n_dims + 1`` points to be well-posed.

    Returns
    -------
    GradientField

    Raises
    ------
    ValueError
        If shapes are inconsistent or there are no source points.
    """
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must be (n, n_dims)")
    n, ndim = positions.shape
    if n == 0:
        raise ValueError("no source points")
    if values.shape != (n,):
        raise ValueError(f"values must be ({n},), got {values.shape}")

    if query_points is None:
        query = positions
        sampled = values
    else:
        query = np.asarray(query_points, dtype=float)
        if query.ndim != 2 or query.shape[1] != ndim:
            raise ValueError(f"query_points must be (m, {ndim})")

    k_eff = min(k, n)
    tree = cKDTree(positions)
    _, idx = tree.query(query, k=k_eff)
    idx = np.atleast_2d(idx.T).T  # ensure (m, k_eff) even when k_eff == 1

    if query_points is not None:
        # Nearest-source value at each query point (consistent with
        # substrate_at's nearest-voxel convention).
        sampled = values[idx[:, 0]]

    gradients = np.zeros((query.shape[0], ndim))
    for i in range(query.shape[0]):
        neigh = idx[i]
        # Local design matrix [1, dx, dy, ...] centered on the query point.
        offsets = positions[neigh] - query[i]
        design = np.column_stack([np.ones(len(neigh)), offsets])
        coeffs, *_ = np.linalg.lstsq(design, values[neigh], rcond=None)
        gradients[i] = coeffs[1:]

    return GradientField(points=query, gradients=gradients, values=sampled)


def substrate_gradient(
    timestep: PhysiCellTimeStep,
    name: str,
    query_points: Optional[np.ndarray] = None,
    k: int = 12,
) -> GradientField:
    """
    Gradient of a substrate concentration field.

    Evaluated at the voxel centers by default. Constant spatial axes (e.g. ``z``
    in a 2-D simulation) are dropped, so a 2-D field yields 2-D gradients.

    Parameters
    ----------
    timestep : PhysiCellTimeStep
        A time step exposing :attr:`voxel_positions` and :attr:`substrates`.
    name : str
        Substrate name.
    query_points : np.ndarray, optional
        Points at which to evaluate, with one column per non-constant axis.
        Defaults to the voxel centers.
    k : int, default 12
        Neighbors per local fit.

    Returns
    -------
    GradientField

    Raises
    ------
    ValueError
        If the substrate is unknown or there is no microenvironment field.
    """
    voxels = timestep.voxel_positions
    substrates = timestep.substrates
    if name not in substrates:
        available = ', '.join(substrates) or '(none)'
        raise ValueError(f"Unknown substrate {name!r}. Available: {available}")
    if voxels.shape[0] == 0:
        raise ValueError("No microenvironment voxels available")

    # Keep only axes that vary, so a planar (2-D) field is treated as 2-D.
    active = np.where(np.ptp(voxels, axis=0) > 0)[0]
    if active.size == 0:
        active = np.array([0])
    positions = voxels[:, active]

    return spatial_gradient(positions, substrates[name], query_points, k=k)


def density_gradient(
    data: SpatialTissueData,
    radius: float,
    query_points: Optional[np.ndarray] = None,
    k: int = 12,
) -> GradientField:
    """
    Gradient of local cell density.

    Local density is the number of neighbors within ``radius`` divided by the
    neighborhood area (2-D) or volume (3-D); its gradient points toward denser
    regions.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Radius defining local density.
    query_points : np.ndarray, optional
        Points at which to evaluate the gradient. Defaults to the cell
        positions.
    k : int, default 12
        Neighbors per local fit.

    Returns
    -------
    GradientField
    """
    coords = data.coordinates
    ndim = coords.shape[1]
    if ndim == 2:
        volume = np.pi * radius ** 2
    else:
        volume = (4.0 / 3.0) * np.pi * radius ** 3

    neighborhoods = compute_neighborhoods(
        data, method='radius', radius=radius, include_self=False
    )
    counts = np.array([len(np.asarray(n)) for n in neighborhoods], dtype=float)
    density = counts / volume

    return spatial_gradient(coords, density, query_points, k=k)
