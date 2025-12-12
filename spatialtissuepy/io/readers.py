"""
Data readers for various file formats.

Supports CSV, JSON, HDF5, and AnnData formats.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd

from spatialtissuepy.core.spatial_data import SpatialTissueData
from spatialtissuepy.core.validators import ValidationError


def read_csv(
    filepath: Union[str, Path],
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: Optional[str] = None,
    celltype_col: str = 'cell_type',
    sample_col: Optional[str] = None,
    marker_cols: Optional[List[str]] = None,
    **read_csv_kwargs
) -> SpatialTissueData:
    """
    Read spatial data from a CSV file.

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
        Column name for sample IDs (for multi-sample data).
    marker_cols : list of str, optional
        Column names for marker expression data. If None, auto-detects
        numeric columns not used for other purposes.
    **read_csv_kwargs
        Additional arguments passed to pandas.read_csv.

    Returns
    -------
    SpatialTissueData
        Loaded spatial data.

    Examples
    --------
    >>> data = read_csv('cells.csv', x_col='X_centroid', y_col='Y_centroid')
    >>> data = read_csv('multi_sample.csv', sample_col='patient_id')
    """
    return SpatialTissueData.from_csv(
        filepath=filepath,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        celltype_col=celltype_col,
        sample_col=sample_col,
        marker_cols=marker_cols,
        **read_csv_kwargs
    )


def read_json(
    filepath: Union[str, Path],
    x_key: str = 'x',
    y_key: str = 'y',
    z_key: Optional[str] = None,
    celltype_key: str = 'cell_type',
    sample_key: Optional[str] = None,
) -> SpatialTissueData:
    """
    Read spatial data from a JSON file.

    Expects JSON format with 'cells' array containing cell objects,
    or a flat array of cell objects.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file.
    x_key, y_key : str
        Keys for X and Y coordinates in each cell object.
    z_key : str, optional
        Key for Z coordinate (for 3D data).
    celltype_key : str
        Key for cell type label.
    sample_key : str, optional
        Key for sample ID.

    Returns
    -------
    SpatialTissueData
        Loaded spatial data.

    Examples
    --------
    JSON format:
    {
        "cells": [
            {"x": 100, "y": 200, "cell_type": "T_cell"},
            {"x": 150, "y": 250, "cell_type": "Tumor"}
        ],
        "metadata": {"tissue": "lung"}
    }
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle nested or flat structure
    if isinstance(data, list):
        cells = data
        metadata = {}
    elif isinstance(data, dict):
        cells = data.get('cells', data.get('data', []))
        metadata = data.get('metadata', {})
        if not cells and all(k in data for k in [x_key, y_key]):
            # Columnar format
            n_cells = len(data[x_key])
            cells = [{k: data[k][i] for k in data if isinstance(data[k], list) and len(data[k]) == n_cells}
                     for i in range(n_cells)]
    else:
        raise ValidationError(f"Unexpected JSON structure: {type(data)}")

    if not cells:
        raise ValidationError("No cells found in JSON file")

    # Extract data
    coords = []
    cell_types = []
    sample_ids = [] if sample_key else None
    marker_data = {}

    for i, cell in enumerate(cells):
        if x_key not in cell or y_key not in cell:
            raise ValidationError(f"Cell {i} missing coordinates")
        
        coord = [cell[x_key], cell[y_key]]
        if z_key and z_key in cell:
            coord.append(cell[z_key])
        coords.append(coord)

        cell_types.append(cell.get(celltype_key, 'unknown'))

        if sample_key:
            sample_ids.append(cell.get(sample_key, 'default'))

        # Collect marker data
        skip_keys = {x_key, y_key, z_key, celltype_key, sample_key}
        for k, v in cell.items():
            if k not in skip_keys and isinstance(v, (int, float)):
                if k not in marker_data:
                    marker_data[k] = []
                marker_data[k].append(v)

    # Convert to arrays
    coordinates = np.array(coords)
    
    markers = None
    if marker_data:
        # Ensure all markers have same length
        expected_len = len(cells)
        marker_data = {k: v for k, v in marker_data.items() if len(v) == expected_len}
        if marker_data:
            markers = pd.DataFrame(marker_data)

    metadata['source_file'] = str(filepath)

    return SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types,
        sample_ids=sample_ids,
        markers=markers,
        metadata=metadata
    )


def read_anndata(
    filepath: Union[str, Path],
    spatial_key: str = 'spatial',
    celltype_key: str = 'cell_type',
    sample_key: Optional[str] = None,
    use_raw: bool = False
) -> SpatialTissueData:
    """
    Read spatial data from an AnnData (h5ad) file.

    Parameters
    ----------
    filepath : str or Path
        Path to .h5ad file.
    spatial_key : str, default 'spatial'
        Key in adata.obsm for spatial coordinates.
    celltype_key : str, default 'cell_type'
        Key in adata.obs for cell type labels.
    sample_key : str, optional
        Key in adata.obs for sample IDs.
    use_raw : bool, default False
        Whether to use adata.raw for expression data.

    Returns
    -------
    SpatialTissueData
        Loaded spatial data.

    Notes
    -----
    Requires anndata package to be installed.
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata package required for read_anndata. "
            "Install with: pip install anndata"
        )

    adata = anndata.read_h5ad(filepath)

    # Get spatial coordinates
    if spatial_key not in adata.obsm:
        raise ValidationError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coordinates = adata.obsm[spatial_key]

    # Get cell types
    if celltype_key not in adata.obs.columns:
        raise ValidationError(
            f"Cell type key '{celltype_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    cell_types = adata.obs[celltype_key].values

    # Get sample IDs
    sample_ids = None
    if sample_key is not None:
        if sample_key not in adata.obs.columns:
            raise ValidationError(f"Sample key '{sample_key}' not found in adata.obs")
        sample_ids = adata.obs[sample_key].values

    # Get expression data
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    markers = pd.DataFrame(X, columns=var_names)

    metadata = {
        'source_file': str(filepath),
        'anndata_uns': dict(adata.uns) if adata.uns else {}
    }

    return SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types,
        sample_ids=sample_ids,
        markers=markers,
        metadata=metadata
    )


def read_hdf5(
    filepath: Union[str, Path],
    coordinates_key: str = 'coordinates',
    cell_types_key: str = 'cell_types',
    sample_ids_key: str = 'sample_ids',
    markers_key: str = 'markers',
    metadata_key: str = 'metadata'
) -> SpatialTissueData:
    """
    Read spatial data from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to .h5 or .hdf5 file.
    coordinates_key : str
        Dataset key for coordinates.
    cell_types_key : str
        Dataset key for cell types.
    sample_ids_key : str
        Dataset key for sample IDs (optional in file).
    markers_key : str
        Group key for marker data.
    metadata_key : str
        Group key for metadata.

    Returns
    -------
    SpatialTissueData
        Loaded spatial data.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py package required for read_hdf5. "
            "Install with: pip install h5py"
        )

    with h5py.File(filepath, 'r') as f:
        # Read coordinates
        if coordinates_key not in f:
            raise ValidationError(f"Coordinates key '{coordinates_key}' not in file")
        coordinates = f[coordinates_key][:]

        # Read cell types
        if cell_types_key not in f:
            raise ValidationError(f"Cell types key '{cell_types_key}' not in file")
        cell_types = f[cell_types_key][:]
        if cell_types.dtype.kind == 'S':  # bytes
            cell_types = np.array([s.decode('utf-8') for s in cell_types])

        # Read sample IDs (optional)
        sample_ids = None
        if sample_ids_key in f:
            sample_ids = f[sample_ids_key][:]
            if sample_ids.dtype.kind == 'S':
                sample_ids = np.array([s.decode('utf-8') for s in sample_ids])

        # Read markers (optional)
        markers = None
        if markers_key in f:
            marker_group = f[markers_key]
            marker_data = {}
            for name in marker_group.keys():
                marker_data[name] = marker_group[name][:]
            markers = pd.DataFrame(marker_data)

        # Read metadata (optional)
        metadata = {'source_file': str(filepath)}
        if metadata_key in f:
            meta_group = f[metadata_key]
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]

    return SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types,
        sample_ids=sample_ids,
        markers=markers,
        metadata=metadata
    )
