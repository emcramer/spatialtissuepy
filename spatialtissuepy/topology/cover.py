"""
Cover construction for Mapper algorithm.

A cover is a collection of overlapping sets that covers the range of the
filter function. The Mapper algorithm clusters points within each cover
element and builds a graph from the overlapping clusters.

Cover Types
-----------
UniformCover : Equal-width intervals with fixed overlap
AdaptiveCover : Equal-count intervals (quantile-based)
BallCover : For 2D filter outputs (spatial covers)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class CoverElement:
    """A single element (interval) of a cover."""
    index: int
    lower: float
    upper: float
    
    def contains(self, value: float) -> bool:
        """Check if value is within this cover element."""
        return self.lower <= value <= self.upper
    
    def __repr__(self) -> str:
        return f"CoverElement({self.index}, [{self.lower:.3f}, {self.upper:.3f}])"


class Cover:
    """Base class for cover constructions."""
    
    def __init__(self):
        self.elements: List[CoverElement] = []
    
    def fit(self, filter_values: np.ndarray) -> 'Cover':
        """
        Fit the cover to filter values.
        
        Parameters
        ----------
        filter_values : np.ndarray
            1D array of filter values.
        
        Returns
        -------
        Cover
            Self, with elements populated.
        """
        raise NotImplementedError
    
    def get_element_members(
        self,
        filter_values: np.ndarray
    ) -> List[np.ndarray]:
        """
        Get point indices belonging to each cover element.
        
        Parameters
        ----------
        filter_values : np.ndarray
            1D array of filter values.
        
        Returns
        -------
        list of np.ndarray
            For each cover element, array of point indices.
        """
        members = []
        for element in self.elements:
            mask = (filter_values >= element.lower) & (filter_values <= element.upper)
            members.append(np.where(mask)[0])
        return members
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __iter__(self):
        return iter(self.elements)
    
    def __getitem__(self, idx: int) -> CoverElement:
        return self.elements[idx]


class UniformCover(Cover):
    """
    Cover with equal-width intervals and fixed overlap.
    
    The standard cover type for Mapper. Divides the filter range into
    n_intervals equal-width intervals, with adjacent intervals overlapping
    by a fraction specified by overlap_fraction.
    
    Parameters
    ----------
    n_intervals : int, default 10
        Number of intervals.
    overlap_fraction : float, default 0.5
        Fraction of interval width that overlaps with neighbors.
        0.5 means 50% overlap (each point typically in ~2 intervals).
    
    Examples
    --------
    >>> cover = UniformCover(n_intervals=10, overlap_fraction=0.5)
    >>> cover.fit(filter_values)
    >>> members = cover.get_element_members(filter_values)
    """
    
    def __init__(
        self,
        n_intervals: int = 10,
        overlap_fraction: float = 0.5
    ):
        super().__init__()
        
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")
        if not 0 <= overlap_fraction < 1:
            raise ValueError("overlap_fraction must be in [0, 1)")
        
        self.n_intervals = n_intervals
        self.overlap_fraction = overlap_fraction
    
    def fit(self, filter_values: np.ndarray) -> 'UniformCover':
        """Fit uniform cover to filter values."""
        filter_values = np.asarray(filter_values)
        
        f_min = filter_values.min()
        f_max = filter_values.max()
        f_range = f_max - f_min
        
        if f_range == 0:
            # All values identical: single interval
            self.elements = [CoverElement(0, f_min - 0.5, f_max + 0.5)]
            return self
        
        # Interval width without overlap
        base_width = f_range / self.n_intervals
        
        # Actual interval width with overlap
        # Each interval extends beyond its base by overlap_fraction/2 on each side
        interval_width = base_width * (1 + self.overlap_fraction)
        
        # Step between interval centers
        step = base_width
        
        self.elements = []
        for i in range(self.n_intervals):
            center = f_min + base_width * (i + 0.5)
            lower = center - interval_width / 2
            upper = center + interval_width / 2
            
            # Ensure we cover the full range
            if i == 0:
                lower = f_min - 1e-10  # Slightly below to include boundary
            if i == self.n_intervals - 1:
                upper = f_max + 1e-10  # Slightly above to include boundary
            
            self.elements.append(CoverElement(i, lower, upper))
        
        return self


class AdaptiveCover(Cover):
    """
    Cover with equal-count intervals (quantile-based).
    
    Instead of equal-width intervals, this cover ensures each interval
    contains approximately the same number of points. This is useful
    when filter values are not uniformly distributed.
    
    Parameters
    ----------
    n_intervals : int, default 10
        Number of intervals.
    overlap_fraction : float, default 0.5
        Fraction of points that overlap with neighboring intervals.
    
    Examples
    --------
    >>> cover = AdaptiveCover(n_intervals=10, overlap_fraction=0.5)
    >>> cover.fit(filter_values)
    """
    
    def __init__(
        self,
        n_intervals: int = 10,
        overlap_fraction: float = 0.5
    ):
        super().__init__()
        
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")
        if not 0 <= overlap_fraction < 1:
            raise ValueError("overlap_fraction must be in [0, 1)")
        
        self.n_intervals = n_intervals
        self.overlap_fraction = overlap_fraction
    
    def fit(self, filter_values: np.ndarray) -> 'AdaptiveCover':
        """Fit adaptive cover using quantiles."""
        filter_values = np.asarray(filter_values)
        n_points = len(filter_values)
        
        if n_points == 0:
            self.elements = []
            return self
        
        # Compute quantiles for interval boundaries
        # Without overlap, boundaries would be at 0, 1/n, 2/n, ..., 1 quantiles
        # With overlap, we extend each interval
        
        base_quantiles = np.linspace(0, 1, self.n_intervals + 1)
        
        # Extend each interval by overlap_fraction / 2 on each side (in quantile space)
        extension = self.overlap_fraction / 2 / self.n_intervals
        
        self.elements = []
        for i in range(self.n_intervals):
            q_lower = max(0, base_quantiles[i] - extension)
            q_upper = min(1, base_quantiles[i + 1] + extension)
            
            lower = np.quantile(filter_values, q_lower)
            upper = np.quantile(filter_values, q_upper)
            
            # Handle edge cases
            if i == 0:
                lower = filter_values.min() - 1e-10
            if i == self.n_intervals - 1:
                upper = filter_values.max() + 1e-10
            
            self.elements.append(CoverElement(i, lower, upper))
        
        return self


class BallCover(Cover):
    """
    Cover using balls in 2D filter space.
    
    For use with 2D filter functions (e.g., combined x and y coordinates).
    Places overlapping balls to cover the 2D filter space.
    
    Note: This is a more advanced cover type for specialized applications.
    
    Parameters
    ----------
    n_balls : int, default 20
        Approximate number of balls.
    overlap_fraction : float, default 0.3
        Fraction of ball radius that overlaps with neighbors.
    """
    
    def __init__(
        self,
        n_balls: int = 20,
        overlap_fraction: float = 0.3
    ):
        super().__init__()
        self.n_balls = n_balls
        self.overlap_fraction = overlap_fraction
        self._centers: Optional[np.ndarray] = None
        self._radius: float = 0.0
    
    def fit(self, filter_values: np.ndarray) -> 'BallCover':
        """
        Fit ball cover to 2D filter values.
        
        Parameters
        ----------
        filter_values : np.ndarray
            2D array of shape (n_points, 2).
        """
        filter_values = np.asarray(filter_values)
        
        if filter_values.ndim != 2 or filter_values.shape[1] != 2:
            raise ValueError("BallCover requires 2D filter values (n_points, 2)")
        
        # Compute bounding box
        mins = filter_values.min(axis=0)
        maxs = filter_values.max(axis=0)
        ranges = maxs - mins
        
        # Estimate grid dimensions
        aspect = ranges[0] / ranges[1] if ranges[1] > 0 else 1
        n_y = int(np.sqrt(self.n_balls / aspect))
        n_x = int(self.n_balls / n_y)
        
        # Ball radius based on grid spacing
        spacing_x = ranges[0] / n_x if n_x > 0 else ranges[0]
        spacing_y = ranges[1] / n_y if n_y > 0 else ranges[1]
        base_radius = max(spacing_x, spacing_y) / 2
        self._radius = base_radius * (1 + self.overlap_fraction)
        
        # Generate ball centers on grid
        centers = []
        for i in range(n_x):
            for j in range(n_y):
                cx = mins[0] + spacing_x * (i + 0.5)
                cy = mins[1] + spacing_y * (j + 0.5)
                centers.append([cx, cy])
        
        self._centers = np.array(centers)
        
        # Create cover elements (using 1D indexing)
        self.elements = [
            CoverElement(i, 0, 0)  # lower/upper not meaningful for balls
            for i in range(len(centers))
        ]
        
        return self
    
    def get_element_members(
        self,
        filter_values: np.ndarray
    ) -> List[np.ndarray]:
        """Get points within each ball."""
        filter_values = np.asarray(filter_values)
        
        members = []
        for center in self._centers:
            distances = np.linalg.norm(filter_values - center, axis=1)
            mask = distances <= self._radius
            members.append(np.where(mask)[0])
        
        return members


def create_cover(
    cover_type: str = 'uniform',
    n_intervals: int = 10,
    overlap_fraction: float = 0.5,
    **kwargs
) -> Cover:
    """
    Factory function to create a cover.
    
    Parameters
    ----------
    cover_type : str, default 'uniform'
        Type of cover: 'uniform', 'adaptive', or 'ball'.
    n_intervals : int, default 10
        Number of intervals/elements.
    overlap_fraction : float, default 0.5
        Overlap fraction.
    **kwargs
        Additional arguments passed to cover constructor.
    
    Returns
    -------
    Cover
        Cover instance (not yet fitted).
    """
    cover_types = {
        'uniform': UniformCover,
        'adaptive': AdaptiveCover,
        'ball': BallCover,
    }
    
    if cover_type not in cover_types:
        raise ValueError(f"Unknown cover type: {cover_type}. "
                        f"Options: {list(cover_types.keys())}")
    
    cls = cover_types[cover_type]
    
    if cover_type == 'ball':
        return cls(n_balls=n_intervals, overlap_fraction=overlap_fraction, **kwargs)
    else:
        return cls(n_intervals=n_intervals, overlap_fraction=overlap_fraction, **kwargs)
