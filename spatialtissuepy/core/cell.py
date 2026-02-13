"""
Lightweight Cell class for representing individual cells.

This class provides a simple interface for accessing cell properties
without the overhead of the full SpatialTissueData container.
"""

from typing import Optional, Dict, Any, List
import numpy as np


class Cell:
    """
    Lightweight representation of a single cell.

    Parameters
    ----------
    cell_id : int
        Unique identifier for the cell.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float, optional
        Z coordinate (for 3D data).
    cell_type : str
        Cell phenotype/type label.
    sample_id : str, optional
        Sample identifier.
    markers : dict, optional
        Dictionary of marker name -> expression value.
    metadata : dict, optional
        Additional cell-level metadata.

    Attributes
    ----------
    cell_id : int
        Unique cell identifier.
    coordinates : np.ndarray
        Coordinate array (2D or 3D).
    cell_type : str
        Cell phenotype label.
    sample_id : str or None
        Sample identifier.
    markers : dict
        Marker expression values.
    metadata : dict
        Additional metadata.
    neighbors : list
        List of neighbor cell IDs (populated by neighborhood analysis).

    Examples
    --------
    >>> cell = Cell(0, x=100.5, y=200.3, cell_type='T_cell')
    >>> cell.coordinates
    array([100.5, 200.3])
    >>> cell.cell_type
    'T_cell'
    """

    __slots__ = [
        'cell_id', '_x', '_y', '_z', 'cell_type', 'sample_id',
        'markers', 'metadata', 'neighbors'
    ]

    def __init__(
        self,
        cell_id: int,
        x: float,
        y: float,
        z: Optional[float] = None,
        cell_type: str = "unknown",
        sample_id: Optional[str] = None,
        markers: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.cell_id = cell_id
        self._x = float(x)
        self._y = float(y)
        self._z = float(z) if z is not None else None
        self.cell_type = str(cell_type)
        self.sample_id = sample_id
        self.markers = markers if markers is not None else {}
        self.metadata = metadata if metadata is not None else {}
        self.neighbors: List[int] = []

    @property
    def x(self) -> float:
        """X coordinate."""
        return self._x

    @property
    def y(self) -> float:
        """Y coordinate."""
        return self._y

    @property
    def z(self) -> Optional[float]:
        """Z coordinate (None for 2D data)."""
        return self._z

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinate array as numpy array."""
        if self._z is not None:
            return np.array([self._x, self._y, self._z])
        return np.array([self._x, self._y])

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return 3 if self._z is not None else 2

    def distance_to(self, other: 'Cell') -> float:
        """
        Calculate Euclidean distance to another cell.

        Parameters
        ----------
        other : Cell
            Another cell object.

        Returns
        -------
        float
            Euclidean distance.

        Raises
        ------
        ValueError
            If cells have different dimensionality.
        """
        if self.ndim != other.ndim:
            raise ValueError(
                f"Cannot compute distance between cells with different dimensionality"
            )
        return float(np.linalg.norm(self.coordinates - other.coordinates))

    def get_marker(self, marker_name: str, default: float = np.nan) -> float:
        """
        Get expression value for a specific marker.

        Parameters
        ----------
        marker_name : str
            Name of the marker.
        default : float, default np.nan
            Value to return if marker not found.

        Returns
        -------
        float
            Marker expression value.
        """
        return self.markers.get(marker_name, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert cell to dictionary representation.

        Returns
        -------
        dict
            Dictionary with all cell properties.
        """
        d = {
            'cell_id': self.cell_id,
            'x': self._x,
            'y': self._y,
            'cell_type': self.cell_type,
        }
        if self._z is not None:
            d['z'] = self._z
        if self.sample_id is not None:
            d['sample_id'] = self.sample_id
        d.update(self.markers)
        d.update(self.metadata)
        return d

    def __repr__(self) -> str:
        coords = f"({self._x:.1f}, {self._y:.1f}"
        if self._z is not None:
            coords += f", {self._z:.1f}"
        coords += ")"
        return f"Cell(id={self.cell_id}, type='{self.cell_type}', pos={coords})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.cell_id == other.cell_id

    def __hash__(self) -> int:
        return hash(self.cell_id)
