"""
Input validation utilities for spatialtissuepy.

Provides validation functions for coordinates, cell types, and data consistency.
"""

from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_coordinates(
    coords: np.ndarray,
    ndim: Optional[int] = None,
    allow_nan: bool = False
) -> np.ndarray:
    """
    Validate coordinate array.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array of shape (n_cells, n_dims).
    ndim : int, optional
        Expected number of dimensions (2 or 3). If None, accepts either.
    allow_nan : bool, default False
        Whether to allow NaN values in coordinates.

    Returns
    -------
    np.ndarray
        Validated coordinate array as float64.

    Raises
    ------
    ValidationError
        If coordinates are invalid.
    """
    coords = np.asarray(coords, dtype=np.float64)
    
    if coords.ndim != 2:
        raise ValidationError(
            f"Coordinates must be 2D array, got {coords.ndim}D"
        )
    
    if coords.shape[0] == 0:
        raise ValidationError("Coordinates array is empty")
    
    if ndim is not None:
        if coords.shape[1] != ndim:
            raise ValidationError(
                f"Expected {ndim}D coordinates, got {coords.shape[1]}D"
            )
    elif coords.shape[1] not in (2, 3):
        raise ValidationError(
            f"Coordinates must be 2D or 3D, got {coords.shape[1]}D"
        )
    
    if not allow_nan and np.any(np.isnan(coords)):
        nan_count = np.sum(np.any(np.isnan(coords), axis=1))
        raise ValidationError(
            f"Coordinates contain {nan_count} rows with NaN values"
        )
    
    if np.any(np.isinf(coords)):
        raise ValidationError("Coordinates contain infinite values")
    
    return coords


def validate_cell_types(
    cell_types: Union[np.ndarray, pd.Series, Sequence],
    n_cells: int
) -> np.ndarray:
    """
    Validate cell type labels.

    Parameters
    ----------
    cell_types : array-like
        Cell type labels for each cell.
    n_cells : int
        Expected number of cells.

    Returns
    -------
    np.ndarray
        Validated cell type array as string dtype.

    Raises
    ------
    ValidationError
        If cell types are invalid.
    """
    cell_types = np.asarray(cell_types, dtype=str)
    
    if cell_types.ndim != 1:
        raise ValidationError(
            f"Cell types must be 1D array, got {cell_types.ndim}D"
        )
    
    if len(cell_types) != n_cells:
        raise ValidationError(
            f"Cell types length ({len(cell_types)}) does not match "
            f"number of cells ({n_cells})"
        )
    
    # Check for empty strings
    empty_mask = cell_types == ""
    if np.any(empty_mask):
        empty_count = np.sum(empty_mask)
        raise ValidationError(
            f"Cell types contain {empty_count} empty strings"
        )
    
    return cell_types


def validate_sample_ids(
    sample_ids: Union[np.ndarray, pd.Series, Sequence, None],
    n_cells: int
) -> Optional[np.ndarray]:
    """
    Validate sample ID labels.

    Parameters
    ----------
    sample_ids : array-like or None
        Sample ID for each cell. If None, returns None.
    n_cells : int
        Expected number of cells.

    Returns
    -------
    np.ndarray or None
        Validated sample ID array as string dtype, or None.

    Raises
    ------
    ValidationError
        If sample IDs are invalid.
    """
    if sample_ids is None:
        return None
    
    sample_ids = np.asarray(sample_ids, dtype=str)
    
    if sample_ids.ndim != 1:
        raise ValidationError(
            f"Sample IDs must be 1D array, got {sample_ids.ndim}D"
        )
    
    if len(sample_ids) != n_cells:
        raise ValidationError(
            f"Sample IDs length ({len(sample_ids)}) does not match "
            f"number of cells ({n_cells})"
        )
    
    return sample_ids


def validate_marker_data(
    markers: Union[np.ndarray, pd.DataFrame, None],
    n_cells: int
) -> Optional[pd.DataFrame]:
    """
    Validate marker expression data.

    Parameters
    ----------
    markers : np.ndarray, pd.DataFrame, or None
        Marker expression matrix. If None, returns None.
    n_cells : int
        Expected number of cells.

    Returns
    -------
    pd.DataFrame or None
        Validated marker data as DataFrame, or None.

    Raises
    ------
    ValidationError
        If marker data is invalid.
    """
    if markers is None:
        return None
    
    if isinstance(markers, np.ndarray):
        if markers.ndim != 2:
            raise ValidationError(
                f"Marker array must be 2D, got {markers.ndim}D"
            )
        markers = pd.DataFrame(
            markers,
            columns=[f"marker_{i}" for i in range(markers.shape[1])]
        )
    elif not isinstance(markers, pd.DataFrame):
        raise ValidationError(
            f"Markers must be np.ndarray or pd.DataFrame, got {type(markers)}"
        )
    
    if len(markers) != n_cells:
        raise ValidationError(
            f"Marker data rows ({len(markers)}) does not match "
            f"number of cells ({n_cells})"
        )
    
    # Check for non-numeric columns
    non_numeric = markers.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValidationError(
            f"Marker data contains non-numeric columns: {non_numeric}"
        )
    
    return markers


def validate_metadata(
    metadata: Union[dict, None]
) -> dict:
    """
    Validate and normalize metadata dictionary.

    Parameters
    ----------
    metadata : dict or None
        Metadata dictionary.

    Returns
    -------
    dict
        Validated metadata dictionary (empty dict if None).
    """
    if metadata is None:
        return {}
    
    if not isinstance(metadata, dict):
        raise ValidationError(
            f"Metadata must be a dictionary, got {type(metadata)}"
        )
    
    return dict(metadata)  # Return a copy


def validate_positive_number(
    value: Union[int, float],
    name: str,
    allow_zero: bool = False
) -> Union[int, float]:
    """
    Validate that a value is a positive number.

    Parameters
    ----------
    value : int or float
        Value to validate.
    name : str
        Name of the parameter (for error messages).
    allow_zero : bool, default False
        Whether to allow zero.

    Returns
    -------
    int or float
        The validated value.

    Raises
    ------
    ValidationError
        If value is not positive (or non-negative if allow_zero).
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    
    return value
