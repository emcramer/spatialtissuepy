"""
Spatial niche identification.

A *niche* (also called a cellular neighborhood) is a recurring local
composition of cell types. Cells are described by the mix of types in their
spatial neighborhood, and those composition vectors are clustered; each cluster
is a niche. This is the composition-clustering approach popularized for
multiplexed imaging (e.g. Schürch/Nolan cellular neighborhoods), and is
complementary to the topic-model view in :mod:`spatialtissuepy.lda`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..core import SpatialTissueData
from ..spatial import neighborhood_composition


@dataclass
class NicheResult:
    """
    Result of :func:`identify_niches`.

    Attributes
    ----------
    labels : np.ndarray
        ``(n_cells,)`` integer niche id for each cell.
    n_niches : int
        Number of niches.
    cell_types : np.ndarray
        ``(n_types,)`` cell-type names, giving the column order of
        :attr:`profiles` and of the composition vectors.
    profiles : np.ndarray
        ``(n_niches, n_types)`` mean neighborhood composition of each niche.
    radius : float or None
        Radius used, if the radius method was chosen.
    k : int or None
        Neighbor count used, if the knn method was chosen.
    """

    labels: np.ndarray
    n_niches: int
    cell_types: np.ndarray
    profiles: np.ndarray
    radius: Optional[float] = None
    k: Optional[int] = None

    def niche_sizes(self) -> Dict[int, int]:
        """Number of cells assigned to each niche."""
        ids, counts = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

    def dominant_types(self, top: int = 3) -> Dict[int, List[Tuple[str, float]]]:
        """
        The most abundant cell types in each niche's mean composition.

        Parameters
        ----------
        top : int, default 3
            How many types to report per niche.

        Returns
        -------
        dict
            ``{niche_id: [(type_name, mean_fraction), ...]}``, largest first.
        """
        result: Dict[int, List[Tuple[str, float]]] = {}
        for niche in range(self.n_niches):
            order = np.argsort(self.profiles[niche])[::-1][:top]
            result[niche] = [
                (str(self.cell_types[j]), float(self.profiles[niche, j]))
                for j in order
            ]
        return result

    def profiles_dataframe(self) -> pd.DataFrame:
        """Niche composition profiles as a DataFrame (niches x types)."""
        return pd.DataFrame(
            self.profiles,
            index=[f'niche_{i}' for i in range(self.n_niches)],
            columns=list(self.cell_types),
        )


def identify_niches(
    data: SpatialTissueData,
    n_niches: int,
    radius: Optional[float] = None,
    k: Optional[int] = None,
    method: str = 'radius',
    include_self: bool = True,
    random_state: Optional[int] = None,
) -> NicheResult:
    """
    Identify spatial niches by clustering neighborhood composition.

    Each cell's local neighborhood composition (the fraction of each cell type
    within its neighborhood) is computed, and cells are grouped into ``n_niches``
    clusters by k-means. A niche is therefore a characteristic local cell-type
    mixture, wherever in the tissue it recurs.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    n_niches : int
        Number of niches (k-means clusters). Must be at least 1 and at most the
        number of cells.
    radius : float, optional
        Neighborhood radius (required when ``method='radius'``).
    k : int, optional
        Number of neighbors (required when ``method='knn'``).
    method : str, default 'radius'
        Neighborhood definition, ``'radius'`` or ``'knn'``.
    include_self : bool, default True
        Whether a cell's own type counts toward its neighborhood composition.
    random_state : int, optional
        Seed for k-means, for reproducibility.

    Returns
    -------
    NicheResult
        Per-cell niche labels and per-niche composition profiles.

    Raises
    ------
    ValueError
        If ``n_niches`` is out of range, or the neighborhood parameter for the
        chosen method is missing.
    """
    if method == 'radius' and radius is None:
        raise ValueError("radius is required when method='radius'")
    if method == 'knn' and k is None:
        raise ValueError("k is required when method='knn'")
    if not 1 <= n_niches <= data.n_cells:
        raise ValueError(
            f"n_niches must be in [1, n_cells={data.n_cells}], got {n_niches}"
        )

    composition = neighborhood_composition(
        data,
        method=method,
        radius=radius,
        k=k,
        include_self=include_self,
        normalize=True,
    )

    kmeans = KMeans(n_clusters=n_niches, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(composition)

    # Per-niche mean composition (from the data, not the k-means centroids, so
    # the profile is a genuine average of member cells even if a niche is empty
    # of some type).
    cell_types = data.cell_types_unique
    profiles = np.zeros((n_niches, len(cell_types)))
    for niche in range(n_niches):
        members = labels == niche
        if np.any(members):
            profiles[niche] = composition[members].mean(axis=0)

    return NicheResult(
        labels=labels.astype(int),
        n_niches=n_niches,
        cell_types=cell_types,
        profiles=profiles,
        radius=radius,
        k=k,
    )
