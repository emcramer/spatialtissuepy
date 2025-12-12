"""
Statistics panel for defining custom metric collections.

A StatisticsPanel defines which metrics to compute and with what parameters.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, 
    Tuple, Union
)
from dataclasses import dataclass, field
import json
import copy

from .registry import get_metric, list_metrics, list_categories, MetricInfo

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


@dataclass
class PanelMetric:
    """
    A metric entry in a panel with its parameters.
    
    Attributes
    ----------
    name : str
        Metric name (from registry) or custom name.
    metric_info : MetricInfo
        Registry info for this metric.
    params : dict
        Parameters to pass to the metric function.
    alias : str, optional
        Optional alias for this metric in the panel.
    """
    name: str
    metric_info: MetricInfo
    params: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None
    
    def compute(self, data: 'SpatialTissueData') -> Dict[str, float]:
        """Compute this metric on data."""
        return self.metric_info(data, **self.params)
    
    @property
    def display_name(self) -> str:
        """Display name (alias if set, else name)."""
        return self.alias or self.name
    
    def __repr__(self) -> str:
        if self.params:
            return f"PanelMetric({self.name!r}, params={self.params})"
        return f"PanelMetric({self.name!r})"


class StatisticsPanel:
    """
    A panel of statistics to compute on spatial tissue data.
    
    Panels define which metrics to compute and with what parameters.
    They can be saved, loaded, and shared.
    
    Parameters
    ----------
    name : str, optional
        Panel name for identification.
    description : str, optional
        Panel description.
    
    Examples
    --------
    >>> panel = StatisticsPanel(name='basic')
    >>> panel.add('cell_counts')
    >>> panel.add('cell_proportions')
    >>> panel.add('mean_nearest_neighbor_distance')
    >>> panel.add('ripleys_k', radii=[50, 100, 200])
    >>> 
    >>> # Compute on data
    >>> from spatialtissuepy.summary import SpatialSummary
    >>> summary = SpatialSummary(data, panel)
    >>> vector = summary.to_array()
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: str = "",
    ):
        self.name = name or "custom"
        self.description = description
        self._metrics: List[PanelMetric] = []
        self._metric_names: set = set()
    
    def add(
        self,
        metric_name: str,
        alias: Optional[str] = None,
        **params
    ) -> 'StatisticsPanel':
        """
        Add a metric to the panel.
        
        Parameters
        ----------
        metric_name : str
            Name of registered metric.
        alias : str, optional
            Alias for this metric instance.
        **params
            Parameters to pass to the metric.
        
        Returns
        -------
        StatisticsPanel
            Self for method chaining.
        
        Examples
        --------
        >>> panel.add('cell_counts')
        >>> panel.add('ripleys_k', radii=[25, 50, 100])
        >>> panel.add('cell_type_ratio', numerator='CD8', denominator='Treg')
        """
        metric_info = get_metric(metric_name)
        
        # Create unique key for this metric instance
        key = alias or metric_name
        if params:
            param_str = '_'.join(f'{k}{v}' for k, v in sorted(params.items()))
            key = f"{key}_{param_str}" if not alias else key
        
        panel_metric = PanelMetric(
            name=metric_name,
            metric_info=metric_info,
            params=params,
            alias=alias,
        )
        
        self._metrics.append(panel_metric)
        self._metric_names.add(key)
        
        return self
    
    def add_all(self, category: Optional[str] = None) -> 'StatisticsPanel':
        """
        Add all registered metrics, optionally filtered by category.
        
        Parameters
        ----------
        category : str, optional
            Only add metrics from this category.
        
        Returns
        -------
        StatisticsPanel
            Self for method chaining.
        """
        for name in list_metrics(category):
            if name not in self._metric_names:
                self.add(name)
        return self
    
    def remove(self, metric_name: str) -> 'StatisticsPanel':
        """
        Remove a metric from the panel.
        
        Parameters
        ----------
        metric_name : str
            Metric name or alias to remove.
        
        Returns
        -------
        StatisticsPanel
            Self for method chaining.
        """
        self._metrics = [
            m for m in self._metrics 
            if m.name != metric_name and m.alias != metric_name
        ]
        self._metric_names.discard(metric_name)
        return self
    
    def clear(self) -> 'StatisticsPanel':
        """Remove all metrics from the panel."""
        self._metrics = []
        self._metric_names = set()
        return self
    
    @property
    def metrics(self) -> List[PanelMetric]:
        """List of metrics in the panel."""
        return self._metrics.copy()
    
    @property
    def n_metrics(self) -> int:
        """Number of metrics in the panel."""
        return len(self._metrics)
    
    def compute(self, data: 'SpatialTissueData') -> Dict[str, float]:
        """
        Compute all metrics on data.
        
        Parameters
        ----------
        data : SpatialTissueData
            Input data.
        
        Returns
        -------
        dict
            Combined results from all metrics.
        """
        results = {}
        
        for metric in self._metrics:
            try:
                metric_results = metric.compute(data)
                results.update(metric_results)
            except Exception as e:
                # Log error but continue with other metrics
                for col in metric.metric_info.returns or [metric.name]:
                    results[col] = float('nan')
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert panel to dictionary for serialization.
        
        Returns
        -------
        dict
            Panel configuration.
        """
        return {
            'name': self.name,
            'description': self.description,
            'metrics': [
                {
                    'name': m.name,
                    'params': m.params,
                    'alias': m.alias,
                }
                for m in self._metrics
            ]
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'StatisticsPanel':
        """
        Create panel from dictionary.
        
        Parameters
        ----------
        config : dict
            Panel configuration from to_dict().
        
        Returns
        -------
        StatisticsPanel
            Reconstructed panel.
        """
        panel = cls(
            name=config.get('name'),
            description=config.get('description', ''),
        )
        
        for metric_config in config.get('metrics', []):
            panel.add(
                metric_config['name'],
                alias=metric_config.get('alias'),
                **metric_config.get('params', {})
            )
        
        return panel
    
    def to_json(self, filepath: str) -> None:
        """Save panel to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'StatisticsPanel':
        """Load panel from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    def copy(self) -> 'StatisticsPanel':
        """Create a copy of this panel."""
        return StatisticsPanel.from_dict(self.to_dict())
    
    def __len__(self) -> int:
        return len(self._metrics)
    
    def __repr__(self) -> str:
        return f"StatisticsPanel({self.name!r}, {len(self)} metrics)"
    
    def __str__(self) -> str:
        lines = [
            f"StatisticsPanel: {self.name}",
            f"  Description: {self.description}" if self.description else "",
            f"  Metrics ({len(self)}):",
        ]
        
        # Group by category
        by_category: Dict[str, List[str]] = {}
        for m in self._metrics:
            cat = m.metric_info.category
            if cat not in by_category:
                by_category[cat] = []
            
            name = m.display_name
            if m.params:
                params_str = ', '.join(f'{k}={v}' for k, v in m.params.items())
                name = f"{name}({params_str})"
            by_category[cat].append(name)
        
        for cat, names in by_category.items():
            lines.append(f"    [{cat}]")
            for name in names:
                lines.append(f"      - {name}")
        
        return '\n'.join(line for line in lines if line)


# ============================================================================
# Predefined Panels
# ============================================================================

def _create_basic_panel() -> StatisticsPanel:
    """Create a basic panel with essential metrics."""
    panel = StatisticsPanel(
        name='basic',
        description='Basic population and spatial statistics'
    )
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('cell_density')
    panel.add('mean_nearest_neighbor_distance')
    panel.add('clark_evans_index')
    panel.add('shannon_diversity')
    return panel


def _create_spatial_panel() -> StatisticsPanel:
    """Create a panel focused on spatial statistics."""
    panel = StatisticsPanel(
        name='spatial',
        description='Comprehensive spatial statistics'
    )
    panel.add('cell_counts')
    panel.add('mean_nearest_neighbor_distance')
    panel.add('clark_evans_index')
    panel.add('ripleys_k', radii=[25, 50, 100, 200])
    panel.add('l_function', radii=[25, 50, 100, 200])
    panel.add('g_function_summary', radii=[10, 25, 50])
    panel.add('convex_hull_metrics')
    return panel


def _create_neighborhood_panel() -> StatisticsPanel:
    """Create a panel focused on neighborhood statistics."""
    panel = StatisticsPanel(
        name='neighborhood',
        description='Neighborhood composition and interaction statistics'
    )
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('mean_neighborhood_entropy', radius=50)
    panel.add('mean_neighborhood_composition', radius=50)
    panel.add('neighborhood_homogeneity', radius=50)
    panel.add('interaction_strength_matrix', radius=50)
    return panel


def _create_comprehensive_panel() -> StatisticsPanel:
    """Create a comprehensive panel with all categories."""
    panel = StatisticsPanel(
        name='comprehensive',
        description='Comprehensive panel including all statistic categories'
    )
    
    # Population
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('cell_density')
    panel.add('shannon_diversity')
    panel.add('simpson_diversity')
    
    # Spatial
    panel.add('mean_nearest_neighbor_distance')
    panel.add('clark_evans_index')
    panel.add('ripleys_k', radii=[50, 100, 200])
    panel.add('g_function_summary', radii=[25, 50])
    panel.add('convex_hull_metrics')
    
    # Neighborhood
    panel.add('mean_neighborhood_entropy', radius=50)
    panel.add('mean_neighborhood_composition', radius=50)
    panel.add('neighborhood_homogeneity', radius=50)
    panel.add('interaction_strength_matrix', radius=50)
    
    # Morphology
    panel.add('spatial_extent')
    panel.add('centroid')
    
    return panel


# Registry of predefined panels
_PREDEFINED_PANELS = {
    'basic': _create_basic_panel,
    'spatial': _create_spatial_panel,
    'neighborhood': _create_neighborhood_panel,
    'comprehensive': _create_comprehensive_panel,
}


def load_panel(name: str) -> StatisticsPanel:
    """
    Load a predefined panel by name.
    
    Parameters
    ----------
    name : str
        Panel name: 'basic', 'spatial', 'neighborhood', 'comprehensive'.
    
    Returns
    -------
    StatisticsPanel
        Copy of the predefined panel.
    """
    if name not in _PREDEFINED_PANELS:
        available = ', '.join(_PREDEFINED_PANELS.keys())
        raise ValueError(f"Unknown panel '{name}'. Available: {available}")
    
    return _PREDEFINED_PANELS[name]()


def list_panels() -> List[str]:
    """List available predefined panels."""
    return list(_PREDEFINED_PANELS.keys())
