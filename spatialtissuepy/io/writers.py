"""
Data writers for various file formats.

Supports CSV, JSON, and HDF5 formats.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import json
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def write_csv(
    data: 'SpatialTissueData',
    filepath: Union[str, Path],
    include_markers: bool = True,
    include_neighborhoods: bool = True,
    **to_csv_kwargs
) -> None:
    """
    Write spatial data to a CSV file.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial data to export.
    filepath : str or Path
        Output file path.
    include_markers : bool, default True
        Whether to include marker expression data.
    include_neighborhoods : bool, default True
        Whether to include neighborhood data (if available).
    **to_csv_kwargs
        Additional arguments passed to DataFrame.to_csv.

    Examples
    --------
    >>> write_csv(data, 'output.csv')
    >>> write_csv(data, 'output.csv', include_markers=False)
    """
    df = data.to_dataframe()

    # Optionally exclude markers
    if not include_markers and data.marker_names is not None:
        df = df.drop(columns=data.marker_names, errors='ignore')

    # Optionally include neighborhoods
    if include_neighborhoods and data.has_neighborhoods:
        neigh_df = pd.DataFrame(
            data.neighborhoods,
            columns=[f"neigh_{ct}" for ct in data.cell_types_unique]
        )
        df = pd.concat([df, neigh_df], axis=1)

    df.to_csv(filepath, index=False, **to_csv_kwargs)


def write_json(
    data: 'SpatialTissueData',
    filepath: Union[str, Path],
    include_markers: bool = True,
    include_metadata: bool = True,
    indent: int = 2
) -> None:
    """
    Write spatial data to a JSON file.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial data to export.
    filepath : str or Path
        Output file path.
    include_markers : bool, default True
        Whether to include marker expression data.
    include_metadata : bool, default True
        Whether to include metadata.
    indent : int, default 2
        JSON indentation level.

    Examples
    --------
    >>> write_json(data, 'output.json')
    """
    output = {
        'cells': [],
        'metadata': data.metadata if include_metadata else {}
    }

    # Add summary info
    output['summary'] = {
        'n_cells': data.n_cells,
        'n_cell_types': data.n_cell_types,
        'n_dims': data.n_dims,
        'cell_types': list(data.cell_types_unique),
        'bounds': data.bounds
    }

    # Build cell list
    coords = data.coordinates
    cell_types = data.cell_types
    sample_ids = data.sample_ids
    markers = data.markers

    for i in range(data.n_cells):
        cell = {
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1]),
            'cell_type': str(cell_types[i])
        }

        if data.n_dims == 3:
            cell['z'] = float(coords[i, 2])

        if sample_ids is not None:
            cell['sample_id'] = str(sample_ids[i])

        if include_markers and markers is not None:
            for col in markers.columns:
                cell[col] = float(markers.iloc[i][col])

        output['cells'].append(cell)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=indent, cls=NumpyEncoder)


def write_hdf5(
    data: 'SpatialTissueData',
    filepath: Union[str, Path],
    compression: str = 'gzip',
    compression_opts: int = 4
) -> None:
    """
    Write spatial data to an HDF5 file.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial data to export.
    filepath : str or Path
        Output file path.
    compression : str, default 'gzip'
        Compression algorithm.
    compression_opts : int, default 4
        Compression level (1-9).

    Examples
    --------
    >>> write_hdf5(data, 'output.h5')
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py package required for write_hdf5. "
            "Install with: pip install h5py"
        )

    with h5py.File(filepath, 'w') as f:
        # Write coordinates
        f.create_dataset(
            'coordinates',
            data=data.coordinates,
            compression=compression,
            compression_opts=compression_opts
        )

        # Write cell types (as bytes for HDF5 compatibility)
        cell_types_bytes = np.array(data.cell_types, dtype='S')
        f.create_dataset(
            'cell_types',
            data=cell_types_bytes,
            compression=compression,
            compression_opts=compression_opts
        )

        # Write sample IDs if present
        if data.sample_ids is not None:
            sample_ids_bytes = np.array(data.sample_ids, dtype='S')
            f.create_dataset(
                'sample_ids',
                data=sample_ids_bytes,
                compression=compression,
                compression_opts=compression_opts
            )

        # Write markers if present
        if data.markers is not None:
            markers_group = f.create_group('markers')
            for col in data.markers.columns:
                markers_group.create_dataset(
                    col,
                    data=data.markers[col].values,
                    compression=compression,
                    compression_opts=compression_opts
                )

        # Write neighborhoods if present
        if data.has_neighborhoods:
            f.create_dataset(
                'neighborhoods',
                data=data.neighborhoods,
                compression=compression,
                compression_opts=compression_opts
            )

        # Write metadata
        meta_group = f.create_group('metadata')
        for key, value in data.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                meta_group.attrs[key] = str(value)


def write_anndata(
    data: 'SpatialTissueData',
    filepath: Union[str, Path],
    spatial_key: str = 'spatial'
) -> None:
    """
    Write spatial data to an AnnData (h5ad) file.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial data to export.
    filepath : str or Path
        Output file path (should end in .h5ad).
    spatial_key : str, default 'spatial'
        Key for spatial coordinates in adata.obsm.

    Examples
    --------
    >>> write_anndata(data, 'output.h5ad')
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata package required for write_anndata. "
            "Install with: pip install anndata"
        )

    # Prepare obs DataFrame
    obs_data = {'cell_type': data.cell_types}
    if data.sample_ids is not None:
        obs_data['sample_id'] = data.sample_ids
    obs = pd.DataFrame(obs_data)

    # Prepare X matrix (markers or empty)
    if data.markers is not None:
        X = data.markers.values
        var = pd.DataFrame(index=data.marker_names)
    else:
        X = np.zeros((data.n_cells, 0))
        var = pd.DataFrame()

    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Add spatial coordinates
    adata.obsm[spatial_key] = data.coordinates

    # Add metadata to uns
    adata.uns['spatialtissuepy_metadata'] = data.metadata

    # Write to file
    adata.write_h5ad(filepath)
