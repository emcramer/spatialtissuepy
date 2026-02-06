"""
Statistics panel for defining custom metric collections.

A StatisticsPanel defines which metrics to compute and with what parameters.
Panels support both registered metrics (from the global registry) and
custom functions added directly to the panel.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional,
    Tuple, Union
)
from dataclasses import dataclass, field
import json
import copy
import functools
import warnings

from .registry import (
    get_metric, list_metrics, list_categories, MetricInfo,
    _validate_metric_function, _validate_metric_output,
    MetricValidationError
)

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


# Type alias for metric functions
MetricFunction = Callable[..., Dict[str, float]]


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
    is_inline : bool
        True if this is an inline custom function (not from registry).
    """
    name: str
    metric_info: MetricInfo
    params: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None
    is_inline: bool = False

    def compute(self, data: 'SpatialTissueData') -> Dict[str, float]:
        """Compute this metric on data."""
        return self.metric_info(data, **self.params)

    @property
    def display_name(self) -> str:
        """Display name (alias if set, else name)."""
        return self.alias or self.name

    @property
    def is_serializable(self) -> bool:
        """Check if this metric can be serialized to JSON."""
        return not self.is_inline

    def __repr__(self) -> str:
        inline_str = ", inline=True" if self.is_inline else ""
        if self.params:
            return f"PanelMetric({self.name!r}, params={self.params}{inline_str})"
        return f"PanelMetric({self.name!r}{inline_str})"


class StatisticsPanel:
    """
    A panel of statistics to compute on spatial tissue data.

    Panels define which metrics to compute and with what parameters.
    They support:
    - Registered metrics from the global registry
    - Custom functions added directly to the panel (inline)

    Parameters
    ----------
    name : str, optional
        Panel name for identification.
    description : str, optional
        Panel description.

    Examples
    --------
    **Basic usage with registered metrics:**

    >>> panel = StatisticsPanel(name='basic')
    >>> panel.add('cell_counts')
    >>> panel.add('cell_proportions')
    >>> panel.add('ripleys_k', radii=[50, 100, 200])

    **Adding custom inline functions:**

    >>> def my_ratio(data):
    ...     counts = data.cell_type_counts
    ...     return {'tumor_stroma_ratio': counts.get('Tumor', 0) / max(counts.get('Stroma', 1), 1)}
    >>>
    >>> panel.add_custom_function('tumor_stroma_ratio', my_ratio)

    **Computing on data:**

    >>> from spatialtissuepy.summary import SpatialSummary
    >>> summary = SpatialSummary(data, panel)
    >>> vector = summary.to_array()

    Notes
    -----
    Custom inline functions (added via `add_custom_function`) are NOT serializable.
    If you need to save/load panels with custom metrics, register them globally
    using `register_custom_metric` first, then add them by name.
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
        Add a registered metric to the panel.

        Parameters
        ----------
        metric_name : str
            Name of registered metric (built-in or custom).
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

        See Also
        --------
        add_custom_function : Add an inline custom function
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
            is_inline=False,
        )

        self._metrics.append(panel_metric)
        self._metric_names.add(key)

        return self

    def add_custom_function(
        self,
        name: str,
        fn: MetricFunction,
        description: str = "",
        validate: bool = True,
        **params
    ) -> 'StatisticsPanel':
        """
        Add a custom function directly to this panel.

        This is a convenient way to add one-off metrics without registering
        them globally. The function is validated and wrapped for error handling.

        Parameters
        ----------
        name : str
            Name for this metric in the panel.
        fn : callable
            Metric function. Must accept SpatialTissueData as first argument
            and return Dict[str, float].
        description : str, optional
            Description of this metric.
        validate : bool, default True
            If True, validate function signature and output.
        **params
            Default parameters to pass to the function.

        Returns
        -------
        StatisticsPanel
            Self for method chaining.

        Raises
        ------
        MetricValidationError
            If function validation fails.
        ValueError
            If a metric with this name already exists in the panel.

        Examples
        --------
        **Simple ratio metric:**

        >>> def cd8_treg_ratio(data):
        ...     counts = data.cell_type_counts
        ...     cd8 = counts.get('CD8_T', 0)
        ...     treg = counts.get('Treg', 0)
        ...     return {'cd8_treg_ratio': cd8 / max(treg, 1)}
        >>>
        >>> panel.add_custom_function('cd8_treg_ratio', cd8_treg_ratio)

        **Custom distance metric:**

        >>> def mean_squared_distance(data):
        ...     from scipy.spatial.distance import pdist
        ...     coords = data.coordinates
        ...     if len(coords) < 2:
        ...         return {'mean_sq_dist': 0.0}
        ...     sq_dists = pdist(coords, metric='sqeuclidean')
        ...     return {'mean_sq_dist': float(sq_dists.mean())}
        >>>
        >>> panel.add_custom_function(
        ...     'mean_squared_distance',
        ...     mean_squared_distance,
        ...     description='Mean squared Euclidean distance between cells'
        ... )

        **With parameters:**

        >>> def cell_type_fraction(data, cell_type='Tumor'):
        ...     counts = data.cell_type_counts
        ...     return {f'{cell_type}_fraction': counts.get(cell_type, 0) / data.n_cells}
        >>>
        >>> panel.add_custom_function(
        ...     'tumor_fraction',
        ...     cell_type_fraction,
        ...     cell_type='Tumor'
        ... )

        Notes
        -----
        Custom inline functions are NOT serializable to JSON. If you need
        persistence, use `register_custom_metric` to register globally first.

        See Also
        --------
        add : Add a registered metric by name
        register_custom_metric : Register metric globally for reuse
        """
        # Check for duplicate name
        if name in self._metric_names:
            raise ValueError(
                f"Metric '{name}' already exists in this panel. "
                f"Use a different name or remove the existing metric first."
            )

        # Validate function
        if validate:
            _validate_metric_function(fn, name, strict=True)

        # Create wrapped function with output validation
        @functools.wraps(fn)
        def validated_fn(data: 'SpatialTissueData', **kwargs) -> Dict[str, float]:
            merged_params = {**params, **kwargs}
            result = fn(data, **merged_params)
            return _validate_metric_output(result, name)

        # Create a MetricInfo for this inline function
        metric_info = MetricInfo(
            name=name,
            func=validated_fn,
            category='inline',
            description=description,
            parameters={},
            returns=[],
            dynamic_columns=True,  # Assume dynamic for custom functions
            is_custom=True,
        )

        panel_metric = PanelMetric(
            name=name,
            metric_info=metric_info,
            params={},  # Params are baked into validated_fn
            alias=None,
            is_inline=True,
        )

        self._metrics.append(panel_metric)
        self._metric_names.add(name)

        return self

    def add_all(
        self,
        category: Optional[str] = None,
        include_custom: bool = True
    ) -> 'StatisticsPanel':
        """
        Add all registered metrics, optionally filtered by category.

        Parameters
        ----------
        category : str, optional
            Only add metrics from this category.
        include_custom : bool, default True
            Whether to include custom registered metrics.

        Returns
        -------
        StatisticsPanel
            Self for method chaining.
        """
        for name in list_metrics(category, include_custom=include_custom):
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

    @property
    def has_inline_metrics(self) -> bool:
        """Check if panel contains inline custom functions."""
        return any(m.is_inline for m in self._metrics)

    @property
    def is_serializable(self) -> bool:
        """Check if entire panel can be serialized to JSON."""
        return not self.has_inline_metrics

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
                import warnings
                warnings.warn(
                    f"Metric '{metric.name}' failed with error: {e}",
                    RuntimeWarning
                )
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

        Raises
        ------
        ValueError
            If panel contains inline custom functions (not serializable).

        Notes
        -----
        Panels with inline custom functions cannot be serialized.
        Use `register_custom_metric` to register metrics globally first.
        """
        if self.has_inline_metrics:
            inline_names = [m.name for m in self._metrics if m.is_inline]
            raise ValueError(
                f"Panel contains inline custom functions that cannot be serialized: "
                f"{inline_names}. Register these metrics globally using "
                f"register_custom_metric() before saving the panel."
            )

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
        """
        Save panel to JSON file.

        Raises
        ------
        ValueError
            If panel contains inline custom functions.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'StatisticsPanel':
        """Load panel from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    def copy(self) -> 'StatisticsPanel':
        """
        Create a copy of this panel.

        Note: Inline custom functions are copied by reference.
        """
        new_panel = StatisticsPanel(
            name=self.name,
            description=self.description,
        )
        new_panel._metrics = self._metrics.copy()
        new_panel._metric_names = self._metric_names.copy()
        return new_panel

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        inline_count = sum(1 for m in self._metrics if m.is_inline)
        if inline_count > 0:
            return f"StatisticsPanel({self.name!r}, {len(self)} metrics, {inline_count} inline)"
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
            if m.is_inline:
                cat = 'inline (custom)'
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
