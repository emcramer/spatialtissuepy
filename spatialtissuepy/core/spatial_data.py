"""
SpatialTissueData: Main data container for spatial tissue analysis.

This module provides the core data structure for storing and manipulating
spatial cell data from multiplexed imaging experiments.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union, Iterator, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatialtissuepy.core.cell import Cell
from spatialtissuepy.core.validators import (
    ValidationError,
    validate_coordinates,
    validate_cell_types,
    validate_sample_ids,
    validate_marker_data,
    validate_metadata,
    validate_positive_number,
)


class SpatialTissueData:
    """
    Container for spatial cell data from tissue samples.

    This class stores cell coordinates, phenotypes, marker expression, and
    metadata for spatial analysis of tissue samples. It supports both single-
    sample and multi-sample datasets.

    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates of shape (n_cells, 2) or (n_cells, 3).
    cell_types : array-like
        Cell type/phenotype labels for each cell.
    sample_ids : array-like, optional
        Sample identifier for each cell (for multi-sample data).
    markers : pd.DataFrame or np.ndarray, optional
        Marker expression data of shape (n_cells, n_markers).
    metadata : dict, optional
        Dataset-level metadata (e.g., tissue type, patient info).
    coordinate_units : str, default 'micrometers'
        Units for coordinates ('micrometers', 'pixels', etc.).

    Attributes
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_dims : int
        Number of spatial dimensions (2 or 3).
    cell_type_counts : pd.Series
        Counts of each cell type.
    sample_ids_unique : np.ndarray
        Unique sample identifiers.
    bounds : dict
        Spatial bounds for each dimension.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.rand(100, 2) * 1000
    >>> types = np.random.choice(['T_cell', 'Tumor', 'Stromal'], 100)
    >>> data = SpatialTissueData(coords, types)
    >>> data.n_cells
    100
    >>> data.cell_type_counts
    T_cell     35
    Tumor      33
    Stromal    32
    dtype: int64
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cell_types: Union[np.ndarray, pd.Series, List[str]],
        sample_ids: Optional[Union[np.ndarray, pd.Series, List[str]]] = None,
        markers: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        coordinate_units: str = "micrometers"
    ):
        # Validate and store coordinates
        self._coordinates = validate_coordinates(coordinates)
        n_cells = self._coordinates.shape[0]

        # Validate and store cell types
        self._cell_types = validate_cell_types(cell_types, n_cells)

        # Validate and store sample IDs
        self._sample_ids = validate_sample_ids(sample_ids, n_cells)

        # Validate and store markers
        self._markers = validate_marker_data(markers, n_cells)

        # Store metadata
        self._metadata = validate_metadata(metadata)
        self._metadata['coordinate_units'] = coordinate_units

        # Initialize caches
        self._kdtree: Optional[cKDTree] = None
        self._neighborhoods: Optional[np.ndarray] = None
        self._neighborhood_params: Optional[Dict] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of cells in the dataset."""
        return self._coordinates.shape[0]

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self._coordinates.shape[1]

    @property
    def coordinates(self) -> np.ndarray:
        """Cell coordinates array (n_cells, n_dims). Read-only copy."""
        return self._coordinates.copy()

    @property
    def cell_types(self) -> np.ndarray:
        """Cell type labels. Read-only copy."""
        return self._cell_types.copy()

    @property
    def sample_ids(self) -> Optional[np.ndarray]:
        """Sample IDs (None if single-sample). Read-only copy."""
        if self._sample_ids is None:
            return None
        return self._sample_ids.copy()

    @property
    def markers(self) -> Optional[pd.DataFrame]:
        """Marker expression DataFrame (None if not provided). Read-only copy."""
        if self._markers is None:
            return None
        return self._markers.copy()

    @property
    def marker_names(self) -> Optional[List[str]]:
        """List of marker names (None if no markers)."""
        if self._markers is None:
            return None
        return list(self._markers.columns)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dataset metadata. Read-only copy."""
        return dict(self._metadata)

    @property
    def coordinate_units(self) -> str:
        """Units for coordinates."""
        return self._metadata.get('coordinate_units', 'micrometers')

    @property
    def cell_type_counts(self) -> pd.Series:
        """Counts of each cell type."""
        unique, counts = np.unique(self._cell_types, return_counts=True)
        return pd.Series(counts, index=unique).sort_values(ascending=False)

    @property
    def cell_types_unique(self) -> np.ndarray:
        """Unique cell type labels."""
        return np.unique(self._cell_types)

    @property
    def n_cell_types(self) -> int:
        """Number of unique cell types."""
        return len(self.cell_types_unique)

    @property
    def sample_ids_unique(self) -> Optional[np.ndarray]:
        """Unique sample identifiers (None if single-sample)."""
        if self._sample_ids is None:
            return None
        return np.unique(self._sample_ids)

    @property
    def n_samples(self) -> int:
        """Number of unique samples (1 if single-sample)."""
        if self._sample_ids is None:
            return 1
        return len(self.sample_ids_unique)

    @property
    def is_multisample(self) -> bool:
        """Whether this is a multi-sample dataset."""
        return self._sample_ids is not None and self.n_samples > 1

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Spatial bounds for each dimension."""
        bounds = {
            'x': (float(self._coordinates[:, 0].min()),
                  float(self._coordinates[:, 0].max())),
            'y': (float(self._coordinates[:, 1].min()),
                  float(self._coordinates[:, 1].max())),
        }
        if self.n_dims == 3:
            bounds['z'] = (float(self._coordinates[:, 2].min()),
                           float(self._coordinates[:, 2].max()))
        return bounds

    @property
    def extent(self) -> Dict[str, float]:
        """Spatial extent (range) for each dimension."""
        b = self.bounds
        extent = {
            'x': b['x'][1] - b['x'][0],
            'y': b['y'][1] - b['y'][0],
        }
        if 'z' in b:
            extent['z'] = b['z'][1] - b['z'][0]
        return extent

    @property
    def kdtree(self) -> cKDTree:
        """
        KD-tree for efficient spatial queries. Built lazily on first access.
        """
        if self._kdtree is None:
            self._kdtree = cKDTree(self._coordinates)
        return self._kdtree

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: Optional[str] = None,
        celltype_col: str = 'cell_type',
        sample_col: Optional[str] = None,
        marker_cols: Optional[List[str]] = None,
        **read_csv_kwargs
    ) -> 'SpatialTissueData':
        """
        Load spatial data from a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to CSV file.
        x_col : str, default 'x'
            Column name for X coordinates.
        y_col : str, default 'y'
            Column name for Y coordinates.
        z_col : str, optional
            Column name for Z coordinates (for 3D data).
        celltype_col : str, default 'cell_type'
            Column name for cell type labels.
        sample_col : str, optional
            Column name for sample IDs.
        marker_cols : list of str, optional
            Column names for marker expression. If None, auto-detects
            numeric columns not used for coordinates.
        **read_csv_kwargs
            Additional arguments passed to pd.read_csv.

        Returns
        -------
        SpatialTissueData
            Loaded spatial data object.

        Examples
        --------
        >>> data = SpatialTissueData.from_csv(
        ...     'cells.csv',
        ...     x_col='X_centroid',
        ...     y_col='Y_centroid',
        ...     celltype_col='phenotype'
        ... )
        """
        df = pd.read_csv(filepath, **read_csv_kwargs)

        # Extract coordinates
        coord_cols = [x_col, y_col]
        if z_col is not None:
            coord_cols.append(z_col)

        for col in coord_cols:
            if col not in df.columns:
                raise ValidationError(f"Coordinate column '{col}' not found")

        coordinates = df[coord_cols].values

        # Extract cell types
        if celltype_col not in df.columns:
            raise ValidationError(f"Cell type column '{celltype_col}' not found")
        cell_types = df[celltype_col].values

        # Extract sample IDs
        sample_ids = None
        if sample_col is not None:
            if sample_col not in df.columns:
                raise ValidationError(f"Sample column '{sample_col}' not found")
            sample_ids = df[sample_col].values

        # Extract markers
        markers = None
        used_cols = set(coord_cols + [celltype_col])
        if sample_col:
            used_cols.add(sample_col)

        if marker_cols is not None:
            missing = set(marker_cols) - set(df.columns)
            if missing:
                raise ValidationError(f"Marker columns not found: {missing}")
            markers = df[marker_cols]
        else:
            # Auto-detect numeric columns
            remaining = [c for c in df.columns if c not in used_cols]
            numeric_cols = df[remaining].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                markers = df[numeric_cols]

        # Store source file in metadata
        metadata = {'source_file': str(filepath)}

        return cls(
            coordinates=coordinates,
            cell_types=cell_types,
            sample_ids=sample_ids,
            markers=markers,
            metadata=metadata
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: Optional[str] = None,
        celltype_col: str = 'cell_type',
        sample_col: Optional[str] = None,
        marker_cols: Optional[List[str]] = None
    ) -> 'SpatialTissueData':
        """
        Create SpatialTissueData from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cell data.
        x_col, y_col, z_col, celltype_col, sample_col, marker_cols
            See from_csv for parameter descriptions.

        Returns
        -------
        SpatialTissueData
            Spatial data object.
        """
        # Extract coordinates
        coord_cols = [x_col, y_col]
        if z_col is not None:
            coord_cols.append(z_col)

        for col in coord_cols:
            if col not in df.columns:
                raise ValidationError(f"Coordinate column '{col}' not found")

        coordinates = df[coord_cols].values
        cell_types = df[celltype_col].values

        sample_ids = None
        if sample_col is not None and sample_col in df.columns:
            sample_ids = df[sample_col].values

        markers = None
        if marker_cols is not None:
            markers = df[marker_cols]

        return cls(
            coordinates=coordinates,
            cell_types=cell_types,
            sample_ids=sample_ids,
            markers=markers
        )

    # -------------------------------------------------------------------------
    # Data Access Methods
    # -------------------------------------------------------------------------

    def get_cell(self, idx: int) -> Cell:
        """
        Get a Cell object for a specific cell index.

        Parameters
        ----------
        idx : int
            Cell index.

        Returns
        -------
        Cell
            Cell object with all properties.
        """
        if idx < 0 or idx >= self.n_cells:
            raise IndexError(f"Cell index {idx} out of range [0, {self.n_cells})")

        coords = self._coordinates[idx]
        z = coords[2] if self.n_dims == 3 else None

        markers = {}
        if self._markers is not None:
            markers = self._markers.iloc[idx].to_dict()

        return Cell(
            cell_id=idx,
            x=coords[0],
            y=coords[1],
            z=z,
            cell_type=self._cell_types[idx],
            sample_id=self._sample_ids[idx] if self._sample_ids is not None else None,
            markers=markers
        )

    def get_cells_by_type(self, cell_type: str) -> np.ndarray:
        """
        Get indices of cells with a specific type.

        Parameters
        ----------
        cell_type : str
            Cell type to select.

        Returns
        -------
        np.ndarray
            Array of cell indices.
        """
        return np.where(self._cell_types == cell_type)[0]

    def get_cells_by_sample(self, sample_id: str) -> np.ndarray:
        """
        Get indices of cells from a specific sample.

        Parameters
        ----------
        sample_id : str
            Sample identifier.

        Returns
        -------
        np.ndarray
            Array of cell indices.
        """
        if self._sample_ids is None:
            raise ValueError("No sample IDs in this dataset")
        return np.where(self._sample_ids == sample_id)[0]

    def subset(
        self,
        indices: Optional[np.ndarray] = None,
        cell_types: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None
    ) -> 'SpatialTissueData':
        """
        Create a subset of the data.

        Parameters
        ----------
        indices : np.ndarray, optional
            Specific cell indices to include.
        cell_types : list of str, optional
            Cell types to include.
        sample_ids : list of str, optional
            Sample IDs to include.

        Returns
        -------
        SpatialTissueData
            Subset of the data.
        """
        mask = np.ones(self.n_cells, dtype=bool)

        if indices is not None:
            idx_mask = np.zeros(self.n_cells, dtype=bool)
            idx_mask[indices] = True
            mask &= idx_mask

        if cell_types is not None:
            type_mask = np.isin(self._cell_types, cell_types)
            mask &= type_mask

        if sample_ids is not None:
            if self._sample_ids is None:
                raise ValueError("Cannot subset by sample_ids: no sample IDs")
            sample_mask = np.isin(self._sample_ids, sample_ids)
            mask &= sample_mask

        return SpatialTissueData(
            coordinates=self._coordinates[mask],
            cell_types=self._cell_types[mask],
            sample_ids=self._sample_ids[mask] if self._sample_ids is not None else None,
            markers=self._markers.iloc[mask] if self._markers is not None else None,
            metadata=self._metadata.copy()
        )

    def subset_sample(self, sample_id: str) -> 'SpatialTissueData':
        """
        Alias for subset(sample_ids=[sample_id]) for backward compatibility.
        """
        return self.subset(sample_ids=[sample_id])

    def iter_cells(self) -> Iterator[Cell]:
        """
        Iterate over all cells as Cell objects.

        Yields
        ------
        Cell
            Cell object for each cell.
        """
        for i in range(self.n_cells):
            yield self.get_cell(i)

    def iter_samples(self) -> Iterator[Tuple[str, 'SpatialTissueData']]:
        """
        Iterate over samples in a multi-sample dataset.

        Yields
        ------
        tuple of (str, SpatialTissueData)
            Sample ID and subset containing only that sample's cells.
        """
        if self._sample_ids is None:
            yield ("default", self)
        else:
            for sid in self.sample_ids_unique:
                yield (sid, self.subset(sample_ids=[sid]))

    # -------------------------------------------------------------------------
    # Spatial Query Methods
    # -------------------------------------------------------------------------

    def query_radius(
        self,
        point: np.ndarray,
        radius: float
    ) -> np.ndarray:
        """
        Find all cells within a radius of a point.

        Parameters
        ----------
        point : np.ndarray
            Query point coordinates.
        radius : float
            Search radius.

        Returns
        -------
        np.ndarray
            Indices of cells within radius.
        """
        radius = validate_positive_number(radius, "radius")
        indices = self.kdtree.query_ball_point(point, radius)
        return np.array(indices, dtype=int)

    def query_knn(
        self,
        point: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to a point.

        Parameters
        ----------
        point : np.ndarray
            Query point coordinates.
        k : int
            Number of neighbors.

        Returns
        -------
        distances : np.ndarray
            Distances to neighbors.
        indices : np.ndarray
            Indices of neighbor cells.
        """
        k = int(validate_positive_number(k, "k"))
        k = min(k, self.n_cells)
        distances, indices = self.kdtree.query(point, k=k)
        return np.atleast_1d(distances), np.atleast_1d(indices)

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordinates, cell types, and markers.
        """
        data = {
            'x': self._coordinates[:, 0],
            'y': self._coordinates[:, 1],
        }
        if self.n_dims == 3:
            data['z'] = self._coordinates[:, 2]

        data['cell_type'] = self._cell_types

        if self._sample_ids is not None:
            data['sample_id'] = self._sample_ids

        df = pd.DataFrame(data)

        if self._markers is not None:
            df = pd.concat([df, self._markers.reset_index(drop=True)], axis=1)

        return df

    def to_csv(self, filepath: Union[str, Path], **kwargs) -> None:
        """
        Export to CSV file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        **kwargs
            Additional arguments passed to DataFrame.to_csv.
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, **kwargs)

    # -------------------------------------------------------------------------
    # Neighborhood Management
    # -------------------------------------------------------------------------

    def add_neighborhoods(
        self,
        neighborhoods: np.ndarray,
        params: Optional[Dict] = None
    ) -> 'SpatialTissueData':
        """
        Add precomputed neighborhood data.

        Parameters
        ----------
        neighborhoods : np.ndarray
            Neighborhood composition matrix (n_cells, n_cell_types).
        params : dict, optional
            Parameters used to compute neighborhoods (for reference).

        Returns
        -------
        SpatialTissueData
            New object with neighborhoods attached (immutable pattern).
        """
        if neighborhoods.shape[0] != self.n_cells:
            raise ValidationError(
                f"Neighborhoods rows ({neighborhoods.shape[0]}) != "
                f"n_cells ({self.n_cells})"
            )

        # Create a copy with neighborhoods
        new_obj = SpatialTissueData(
            coordinates=self._coordinates,
            cell_types=self._cell_types,
            sample_ids=self._sample_ids,
            markers=self._markers,
            metadata=self._metadata.copy()
        )
        new_obj._neighborhoods = neighborhoods
        new_obj._neighborhood_params = params
        return new_obj

    @property
    def neighborhoods(self) -> Optional[np.ndarray]:
        """Neighborhood composition matrix (None if not computed)."""
        return self._neighborhoods

    @property
    def has_neighborhoods(self) -> bool:
        """Whether neighborhoods have been computed."""
        return self._neighborhoods is not None

    # -------------------------------------------------------------------------
    # Dunder Methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_cells

    def __repr__(self) -> str:
        sample_str = f", {self.n_samples} samples" if self.is_multisample else ""
        marker_str = f", {len(self.marker_names)} markers" if self._markers is not None else ""
        return (
            f"SpatialTissueData({self.n_cells} cells, "
            f"{self.n_cell_types} cell types{sample_str}{marker_str})"
        )

    def __str__(self) -> str:
        lines = [
            f"SpatialTissueData",
            f"  Cells: {self.n_cells}",
            f"  Dimensions: {self.n_dims}D",
            f"  Cell types: {self.n_cell_types}",
        ]
        if self.is_multisample:
            lines.append(f"  Samples: {self.n_samples}")
        if self._markers is not None:
            lines.append(f"  Markers: {len(self.marker_names)}")
        lines.append(f"  Bounds: x=[{self.bounds['x'][0]:.1f}, {self.bounds['x'][1]:.1f}], "
                     f"y=[{self.bounds['y'][0]:.1f}, {self.bounds['y'][1]:.1f}]")
        return "\n".join(lines)
