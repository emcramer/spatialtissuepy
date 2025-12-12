"""
Metric registry for spatial statistics summary.

Provides a decorator-based registration system for metrics, enabling
discovery and documentation of available statistics.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, 
    Type, Union
)
from dataclasses import dataclass, field
import functools

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


# Type alias for metric functions
MetricFunction = Callable[..., Dict[str, float]]


@dataclass
class MetricInfo:
    """
    Metadata about a registered metric.
    
    Attributes
    ----------
    name : str
        Unique metric identifier.
    func : callable
        The metric function.
    category : str
        Metric category (population, spatial, neighborhood, etc.).
    description : str
        Human-readable description.
    parameters : dict
        Parameter names and types.
    returns : list of str
        Names of columns this metric returns (can be dynamic).
    """
    name: str
    func: MetricFunction
    category: str
    description: str = ""
    parameters: Dict[str, Type] = field(default_factory=dict)
    returns: List[str] = field(default_factory=list)
    dynamic_columns: bool = False  # True if column names depend on data
    
    def __call__(
        self,
        data: 'SpatialTissueData',
        **kwargs
    ) -> Dict[str, float]:
        """Call the metric function."""
        return self.func(data, **kwargs)
    
    def __repr__(self) -> str:
        return f"MetricInfo(name={self.name!r}, category={self.category!r})"


class MetricRegistry:
    """
    Registry for spatial statistics metrics.
    
    Provides metric discovery, documentation, and access.
    """
    
    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        category: str = 'other',
        description: str = "",
        parameters: Optional[Dict[str, Type]] = None,
        returns: Optional[List[str]] = None,
        dynamic_columns: bool = False,
    ) -> Callable[[MetricFunction], MetricFunction]:
        """
        Decorator to register a metric function.
        
        Parameters
        ----------
        name : str
            Unique metric identifier.
        category : str
            Metric category.
        description : str
            Human-readable description.
        parameters : dict, optional
            Parameter names and types.
        returns : list of str, optional
            Column names this metric returns.
        dynamic_columns : bool
            Whether column names depend on data.
        
        Returns
        -------
        callable
            Decorator function.
        
        Examples
        --------
        >>> @registry.register(
        ...     name='cell_counts',
        ...     category='population',
        ...     description='Count of cells per type',
        ...     dynamic_columns=True
        ... )
        ... def cell_counts(data):
        ...     return {'n_cells': data.n_cells, ...}
        """
        def decorator(func: MetricFunction) -> MetricFunction:
            info = MetricInfo(
                name=name,
                func=func,
                category=category,
                description=description,
                parameters=parameters or {},
                returns=returns or [],
                dynamic_columns=dynamic_columns,
            )
            
            self._metrics[name] = info
            
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def get(self, name: str) -> MetricInfo:
        """
        Get a registered metric by name.
        
        Parameters
        ----------
        name : str
            Metric name.
        
        Returns
        -------
        MetricInfo
            Metric information and function.
        
        Raises
        ------
        KeyError
            If metric not found.
        """
        if name not in self._metrics:
            available = ', '.join(sorted(self._metrics.keys()))
            raise KeyError(
                f"Metric '{name}' not found. Available: {available}"
            )
        return self._metrics[name]
    
    def list_metrics(
        self,
        category: Optional[str] = None
    ) -> List[str]:
        """
        List available metrics.
        
        Parameters
        ----------
        category : str, optional
            Filter by category.
        
        Returns
        -------
        list of str
            Metric names.
        """
        if category is not None:
            return self._categories.get(category, []).copy()
        return list(self._metrics.keys())
    
    def list_categories(self) -> List[str]:
        """List available metric categories."""
        return list(self._categories.keys())
    
    def describe(self, name: str) -> str:
        """Get description of a metric."""
        info = self.get(name)
        lines = [
            f"Metric: {info.name}",
            f"Category: {info.category}",
            f"Description: {info.description}",
        ]
        if info.parameters:
            lines.append(f"Parameters: {info.parameters}")
        if info.returns:
            lines.append(f"Returns: {info.returns}")
        return "\n".join(lines)
    
    def __contains__(self, name: str) -> bool:
        return name in self._metrics
    
    def __len__(self) -> int:
        return len(self._metrics)
    
    def __repr__(self) -> str:
        return f"MetricRegistry({len(self)} metrics in {len(self._categories)} categories)"


# Global registry instance
_registry = MetricRegistry()


def register_metric(
    name: str,
    category: str = 'other',
    description: str = "",
    parameters: Optional[Dict[str, Type]] = None,
    returns: Optional[List[str]] = None,
    dynamic_columns: bool = False,
) -> Callable[[MetricFunction], MetricFunction]:
    """
    Decorator to register a metric in the global registry.
    
    See MetricRegistry.register for parameters.
    """
    return _registry.register(
        name=name,
        category=category,
        description=description,
        parameters=parameters,
        returns=returns,
        dynamic_columns=dynamic_columns,
    )


def get_metric(name: str) -> MetricInfo:
    """Get a metric from the global registry."""
    return _registry.get(name)


def list_metrics(category: Optional[str] = None) -> List[str]:
    """List metrics in the global registry."""
    return _registry.list_metrics(category)


def list_categories() -> List[str]:
    """List metric categories in the global registry."""
    return _registry.list_categories()


def get_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _registry
