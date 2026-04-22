"""
Metric registry for spatial statistics summary.

Provides a decorator-based registration system for metrics, enabling
discovery and documentation of available statistics.

Custom Metrics
--------------
Users can register custom metrics using either:

1. Decorator pattern (recommended for reusable metrics):
   >>> @register_custom_metric(
   ...     name='my_ratio',
   ...     description='Custom cell type ratio'
   ... )
   ... def my_ratio(data: SpatialTissueData) -> Dict[str, float]:
   ...     counts = data.cell_type_counts
   ...     return {'my_ratio': counts.get('A', 0) / max(counts.get('B', 1), 1)}

2. Direct registration (for quick one-off metrics):
   >>> def quick_metric(data):
   ...     return {'value': data.n_cells * 2}
   >>> register_custom_metric(
   ...     name='quick_metric',
   ...     fn=quick_metric,
   ...     description='A quick custom metric'
   ... )
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional,
    Type, Union, get_type_hints
)
from dataclasses import dataclass, field
import functools
import inspect
import warnings

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


# Type alias for metric functions
MetricFunction = Callable[..., Dict[str, float]]


class MetricValidationError(Exception):
    """Raised when a metric function fails validation."""
    pass


class MetricRegistrationError(Exception):
    """Raised when metric registration fails."""
    pass


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
        Metric category (population, spatial, neighborhood, custom, etc.).
    description : str
        Human-readable description.
    parameters : dict
        Parameter names and types.
    returns : list of str
        Names of columns this metric returns (can be dynamic).
    dynamic_columns : bool
        True if column names depend on data.
    is_custom : bool
        True if this is a user-defined custom metric.
    required_dependencies : list of str
        Module names required for this metric.
    """
    name: str
    func: MetricFunction
    category: str
    description: str = ""
    parameters: Dict[str, Type] = field(default_factory=dict)
    returns: List[str] = field(default_factory=list)
    dynamic_columns: bool = False
    is_custom: bool = False
    required_dependencies: List[str] = field(default_factory=list)

    def __call__(
        self,
        data: 'SpatialTissueData',
        **kwargs
    ) -> Dict[str, float]:
        """Call the metric function."""
        return self.func(data, **kwargs)

    def __repr__(self) -> str:
        custom_str = ", custom=True" if self.is_custom else ""
        return f"MetricInfo(name={self.name!r}, category={self.category!r}{custom_str})"

    def __reduce__(self):
        """Pickle by registry name, not by function reference.

        The ``func`` attribute is set during registration to the unwrapped
        metric function, but the decorator returns a ``functools.wraps``
        wrapper as the module attribute. Pickle resolves functions by
        ``__qualname__``, so it finds the wrapper at that name and refuses
        to serialise the stored unwrapped function ("not the same object as
        ...") -- which is the root cause of the long-standing panel
        serialisation failure.

        Instead, we pickle a MetricInfo as its name and re-resolve from the
        global registry on unpickle. That's safe for any registered metric
        (built-in or custom) as long as the registry is populated in the
        unpickling process -- which it is for built-ins at import time.
        Custom metrics must be re-registered with ``register_custom_metric``
        before loading; otherwise unpickling raises a clear error
        identifying which metric is missing.

        Inline (per-panel) MetricInfo objects do not live in the registry
        and cannot be pickled -- they raise a ``TypeError`` here, which is
        the correct behavior: panels with inline functions have always been
        flagged as non-JSON-serialisable, and pickling them would produce
        objects that can't be loaded in a fresh process.
        """
        from spatialtissuepy.summary.registry import _resolve_metric_for_pickle

        registered = _registry._metrics.get(self.name)
        if registered is None:
            raise TypeError(
                f"MetricInfo {self.name!r} is not in the global registry "
                "and cannot be pickled. Inline metrics added via "
                "StatisticsPanel.add_custom_function() are per-panel only; "
                "register them globally with register_custom_metric() "
                "before saving the containing panel."
            )
        return (_resolve_metric_for_pickle, (self.name,))


def _validate_metric_function(
    func: Callable,
    name: str,
    strict: bool = True
) -> None:
    """
    Validate that a function conforms to the metric interface.

    Parameters
    ----------
    func : callable
        Function to validate.
    name : str
        Metric name (for error messages).
    strict : bool
        If True, enforce type hints and signature requirements.

    Raises
    ------
    MetricValidationError
        If validation fails.
    """
    if not callable(func):
        raise MetricValidationError(
            f"Metric '{name}': Expected callable, got {type(func).__name__}"
        )

    # Check signature
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        raise MetricValidationError(
            f"Metric '{name}': Cannot inspect function signature: {e}"
        )

    params = list(sig.parameters.values())

    # Must have at least one parameter (data)
    if len(params) == 0:
        raise MetricValidationError(
            f"Metric '{name}': Function must accept at least one parameter "
            "(the SpatialTissueData object)"
        )

    # First parameter should accept SpatialTissueData
    first_param = params[0]
    if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
        raise MetricValidationError(
            f"Metric '{name}': First parameter cannot be *args. "
            "Expected a named parameter for SpatialTissueData."
        )

    if strict:
        # Check for type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Check return type hint if available
        if 'return' in hints:
            return_hint = hints['return']
            # Allow Dict[str, float], dict, or Any
            hint_str = str(return_hint)
            if 'Dict' not in hint_str and 'dict' not in hint_str:
                warnings.warn(
                    f"Metric '{name}': Return type hint should be Dict[str, float], "
                    f"got {return_hint}. Metric may still work if it returns the correct type.",
                    UserWarning
                )


def _validate_metric_output(
    result: Any,
    name: str
) -> Dict[str, float]:
    """
    Validate the output of a metric function.

    Parameters
    ----------
    result : Any
        Output from metric function.
    name : str
        Metric name (for error messages).

    Returns
    -------
    Dict[str, float]
        Validated result.

    Raises
    ------
    MetricValidationError
        If output is invalid.
    """
    if not isinstance(result, dict):
        raise MetricValidationError(
            f"Metric '{name}': Expected Dict[str, float], got {type(result).__name__}"
        )

    validated = {}
    for key, value in result.items():
        if not isinstance(key, str):
            raise MetricValidationError(
                f"Metric '{name}': Dictionary keys must be strings, "
                f"got {type(key).__name__} for key {key!r}"
            )

        try:
            validated[key] = float(value)
        except (TypeError, ValueError) as e:
            raise MetricValidationError(
                f"Metric '{name}': Value for key '{key}' must be numeric, "
                f"got {type(value).__name__}: {e}"
            )

    return validated


def _check_dependencies(
    dependencies: List[str],
    name: str
) -> None:
    """
    Check that required dependencies are available.

    Parameters
    ----------
    dependencies : list of str
        Module names to check.
    name : str
        Metric name (for error messages).

    Raises
    ------
    MetricRegistrationError
        If a dependency is not available.
    """
    import importlib

    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        raise MetricRegistrationError(
            f"Metric '{name}' requires the following dependencies that are not "
            f"installed: {', '.join(missing)}. Please install them with pip."
        )


class MetricRegistry:
    """
    Registry for spatial statistics metrics.

    Provides metric discovery, documentation, and access.
    Supports both built-in and user-defined custom metrics.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._categories: Dict[str, List[str]] = {}
        self._custom_metrics: set = set()  # Track custom metric names

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
                is_custom=False,
            )

            self._metrics[name] = info

            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def register_custom(
        self,
        name: str,
        fn: Optional[MetricFunction] = None,
        category: str = 'custom',
        description: str = "",
        parameters: Optional[Dict[str, Type]] = None,
        returns: Optional[List[str]] = None,
        dynamic_columns: bool = False,
        required_dependencies: Optional[List[str]] = None,
        overwrite: bool = False,
        validate: bool = True,
    ) -> Union[Callable[[MetricFunction], MetricFunction], None]:
        """
        Register a custom user-defined metric.

        Can be used as a decorator or called directly with a function.

        Parameters
        ----------
        name : str
            Unique metric identifier.
        fn : callable, optional
            Metric function. If None, returns a decorator.
        category : str, default 'custom'
            Metric category.
        description : str
            Human-readable description.
        parameters : dict, optional
            Parameter names and types for documentation.
        returns : list of str, optional
            Column names this metric returns.
        dynamic_columns : bool, default False
            Whether column names depend on data.
        required_dependencies : list of str, optional
            Module names required for this metric.
        overwrite : bool, default False
            If True, allow overwriting existing metrics.
        validate : bool, default True
            If True, validate function signature and type hints.

        Returns
        -------
        callable or None
            If fn is None, returns decorator. Otherwise returns None.

        Raises
        ------
        MetricRegistrationError
            If metric name exists and overwrite=False, or dependencies missing.
        MetricValidationError
            If function validation fails.

        Examples
        --------
        As a decorator:

        >>> @register_custom_metric(
        ...     name='cd8_treg_ratio',
        ...     description='Ratio of CD8+ T cells to Tregs'
        ... )
        ... def cd8_treg_ratio(data: SpatialTissueData) -> Dict[str, float]:
        ...     counts = data.cell_type_counts
        ...     cd8 = counts.get('CD8_T', 0)
        ...     treg = counts.get('Treg', 0)
        ...     ratio = cd8 / treg if treg > 0 else float('inf')
        ...     return {'cd8_treg_ratio': ratio}

        Direct registration:

        >>> def my_metric(data):
        ...     return {'value': data.n_cells}
        >>> register_custom_metric(name='my_metric', fn=my_metric)
        """
        def _register(func: MetricFunction) -> MetricFunction:
            # Check for existing metric
            if name in self._metrics and not overwrite:
                existing = self._metrics[name]
                if existing.is_custom:
                    raise MetricRegistrationError(
                        f"Custom metric '{name}' already exists. "
                        f"Use overwrite=True to replace it."
                    )
                else:
                    raise MetricRegistrationError(
                        f"Cannot overwrite built-in metric '{name}'. "
                        f"Choose a different name for your custom metric."
                    )

            # Check dependencies
            deps = required_dependencies or []
            if deps:
                _check_dependencies(deps, name)

            # Validate function
            if validate:
                _validate_metric_function(func, name, strict=True)

            # Create wrapped function with output validation
            @functools.wraps(func)
            def validated_func(data: 'SpatialTissueData', **kwargs) -> Dict[str, float]:
                result = func(data, **kwargs)
                return _validate_metric_output(result, name)

            # Create MetricInfo
            info = MetricInfo(
                name=name,
                func=validated_func,
                category=category,
                description=description,
                parameters=parameters or {},
                returns=returns or [],
                dynamic_columns=dynamic_columns,
                is_custom=True,
                required_dependencies=deps,
            )

            # Register
            self._metrics[name] = info
            self._custom_metrics.add(name)

            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)

            return func

        # Handle both decorator and direct call
        if fn is not None:
            _register(fn)
            return None
        else:
            return _register

    def unregister_custom(self, name: str) -> bool:
        """
        Remove a custom metric from the registry.

        Parameters
        ----------
        name : str
            Metric name to remove.

        Returns
        -------
        bool
            True if metric was removed, False if not found.

        Raises
        ------
        MetricRegistrationError
            If attempting to remove a built-in metric.
        """
        if name not in self._metrics:
            return False

        if name not in self._custom_metrics:
            raise MetricRegistrationError(
                f"Cannot remove built-in metric '{name}'. "
                f"Only custom metrics can be unregistered."
            )

        info = self._metrics.pop(name)
        self._custom_metrics.discard(name)

        # Remove from category list
        if info.category in self._categories:
            self._categories[info.category] = [
                m for m in self._categories[info.category] if m != name
            ]
            # Clean up empty category
            if not self._categories[info.category]:
                del self._categories[info.category]

        return True

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
        category: Optional[str] = None,
        include_custom: bool = True
    ) -> List[str]:
        """
        List available metrics.

        Parameters
        ----------
        category : str, optional
            Filter by category.
        include_custom : bool, default True
            Whether to include custom metrics.

        Returns
        -------
        list of str
            Metric names.
        """
        if category is not None:
            metrics = self._categories.get(category, []).copy()
        else:
            metrics = list(self._metrics.keys())

        if not include_custom:
            metrics = [m for m in metrics if m not in self._custom_metrics]

        return metrics

    def list_custom_metrics(self) -> List[str]:
        """List only custom user-defined metrics."""
        return list(self._custom_metrics)

    def list_categories(self) -> List[str]:
        """List available metric categories."""
        return list(self._categories.keys())

    def is_custom(self, name: str) -> bool:
        """Check if a metric is custom (user-defined)."""
        return name in self._custom_metrics

    def describe(self, name: str) -> str:
        """Get description of a metric."""
        info = self.get(name)
        lines = [
            f"Metric: {info.name}",
            f"Category: {info.category}",
            f"Custom: {info.is_custom}",
            f"Description: {info.description}",
        ]
        if info.parameters:
            lines.append(f"Parameters: {info.parameters}")
        if info.returns:
            lines.append(f"Returns: {info.returns}")
        if info.required_dependencies:
            lines.append(f"Dependencies: {info.required_dependencies}")
        return "\n".join(lines)

    def clear_custom_metrics(self) -> int:
        """
        Remove all custom metrics from the registry.

        Returns
        -------
        int
            Number of metrics removed.
        """
        count = 0
        for name in list(self._custom_metrics):
            if self.unregister_custom(name):
                count += 1
        return count

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        n_custom = len(self._custom_metrics)
        n_builtin = len(self._metrics) - n_custom
        return (
            f"MetricRegistry({n_builtin} built-in, {n_custom} custom metrics "
            f"in {len(self._categories)} categories)"
        )


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
    Decorator to register a built-in metric in the global registry.

    This is intended for internal use by the spatialtissuepy package.
    For user-defined metrics, use register_custom_metric() instead.

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


def register_custom_metric(
    name: str,
    fn: Optional[MetricFunction] = None,
    category: str = 'custom',
    description: str = "",
    parameters: Optional[Dict[str, Type]] = None,
    returns: Optional[List[str]] = None,
    dynamic_columns: bool = False,
    required_dependencies: Optional[List[str]] = None,
    overwrite: bool = False,
    validate: bool = True,
) -> Union[Callable[[MetricFunction], MetricFunction], None]:
    """
    Register a custom user-defined metric in the global registry.

    Can be used as a decorator or called directly with a function.
    Once registered, the metric is available session-wide and can be
    added to any StatisticsPanel by name.

    Parameters
    ----------
    name : str
        Unique metric identifier. Will be used to add the metric to panels.
    fn : callable, optional
        Metric function. If None, returns a decorator.
        Function must accept a SpatialTissueData object as first argument
        and return Dict[str, float].
    category : str, default 'custom'
        Metric category for organization. Common categories:
        'custom', 'population', 'spatial', 'neighborhood', 'interaction'.
    description : str
        Human-readable description of what this metric computes.
    parameters : dict, optional
        Parameter names and types for documentation.
        Example: {'radius': float, 'cell_type': str}
    returns : list of str, optional
        Column names this metric returns (if known in advance).
    dynamic_columns : bool, default False
        Set True if column names depend on the data (e.g., one per cell type).
    required_dependencies : list of str, optional
        Module names required for this metric. Registration will fail
        if any dependency is not installed.
    overwrite : bool, default False
        If True, allow overwriting existing custom metrics.
        Cannot overwrite built-in metrics.
    validate : bool, default True
        If True, validate function signature and output format.

    Returns
    -------
    callable or None
        If fn is None, returns decorator. Otherwise returns None.

    Raises
    ------
    MetricRegistrationError
        If metric name exists and overwrite=False, or dependencies missing.
    MetricValidationError
        If function validation fails.

    Examples
    --------
    **Example 1: Decorator pattern (recommended for reusable metrics)**

    >>> from spatialtissuepy.summary import register_custom_metric
    >>> from spatialtissuepy import SpatialTissueData
    >>> from typing import Dict
    >>>
    >>> @register_custom_metric(
    ...     name='cd8_treg_ratio',
    ...     category='interaction',
    ...     description='Ratio of CD8+ T cells to regulatory T cells'
    ... )
    ... def cd8_treg_ratio(data: SpatialTissueData) -> Dict[str, float]:
    ...     counts = data.cell_type_counts
    ...     cd8 = counts.get('CD8_T', 0)
    ...     treg = counts.get('Treg', 0)
    ...     ratio = cd8 / treg if treg > 0 else float('inf')
    ...     return {'cd8_treg_ratio': ratio}

    **Example 2: Direct registration (for quick metrics)**

    >>> def tumor_immune_ratio(data):
    ...     counts = data.cell_type_counts
    ...     tumor = counts.get('Tumor', 0)
    ...     immune = sum(counts.get(t, 0) for t in ['CD8_T', 'CD4_T', 'Treg'])
    ...     return {'tumor_immune_ratio': tumor / max(immune, 1)}
    >>>
    >>> register_custom_metric(
    ...     name='tumor_immune_ratio',
    ...     fn=tumor_immune_ratio,
    ...     description='Ratio of tumor cells to total immune cells'
    ... )

    **Example 3: Metric with dependencies**

    >>> @register_custom_metric(
    ...     name='custom_distance_metric',
    ...     required_dependencies=['scipy'],
    ...     description='Custom squared Euclidean distance metric'
    ... )
    ... def custom_distance_metric(data: SpatialTissueData) -> Dict[str, float]:
    ...     from scipy.spatial.distance import pdist
    ...     coords = data.coordinates
    ...     sq_dists = pdist(coords, metric='sqeuclidean')
    ...     return {
    ...         'mean_sq_dist': float(sq_dists.mean()),
    ...         'max_sq_dist': float(sq_dists.max())
    ...     }

    **Using custom metrics in panels**

    >>> from spatialtissuepy.summary import StatisticsPanel
    >>> panel = StatisticsPanel()
    >>> panel.add('cd8_treg_ratio')  # Use registered name
    >>> panel.add('tumor_immune_ratio')

    See Also
    --------
    unregister_custom_metric : Remove a custom metric
    list_custom_metrics : List all custom metrics
    StatisticsPanel.add_custom_function : Add metric directly to panel
    """
    return _registry.register_custom(
        name=name,
        fn=fn,
        category=category,
        description=description,
        parameters=parameters,
        returns=returns,
        dynamic_columns=dynamic_columns,
        required_dependencies=required_dependencies,
        overwrite=overwrite,
        validate=validate,
    )


def unregister_custom_metric(name: str) -> bool:
    """
    Remove a custom metric from the global registry.

    Parameters
    ----------
    name : str
        Metric name to remove.

    Returns
    -------
    bool
        True if metric was removed, False if not found.

    Raises
    ------
    MetricRegistrationError
        If attempting to remove a built-in metric.

    Examples
    --------
    >>> unregister_custom_metric('my_custom_metric')
    True
    """
    return _registry.unregister_custom(name)


def list_custom_metrics() -> List[str]:
    """
    List all custom user-defined metrics in the global registry.

    Returns
    -------
    list of str
        Names of custom metrics.

    Examples
    --------
    >>> list_custom_metrics()
    ['cd8_treg_ratio', 'tumor_immune_ratio', 'custom_distance_metric']
    """
    return _registry.list_custom_metrics()


def clear_custom_metrics() -> int:
    """
    Remove all custom metrics from the global registry.

    This is useful for cleaning up between analysis sessions or tests.

    Returns
    -------
    int
        Number of metrics removed.

    Examples
    --------
    >>> clear_custom_metrics()
    3
    """
    return _registry.clear_custom_metrics()


def get_metric(name: str) -> MetricInfo:
    """Get a metric from the global registry."""
    return _registry.get(name)


def _resolve_metric_for_pickle(name: str) -> 'MetricInfo':
    """Unpickle hook for MetricInfo: fetch the live registry entry by name.

    Raises a descriptive error when the name is not registered in the
    current process (e.g. a custom metric that was not re-registered
    before loading a saved panel). Kept as a module-level function so
    pickle can import it by qualname.
    """
    try:
        return _registry.get(name)
    except KeyError as exc:
        raise RuntimeError(
            f"Cannot unpickle MetricInfo {name!r}: no metric with that "
            "name is registered in the current process. Built-in metrics "
            "register automatically at import time; custom metrics must "
            "be re-registered with register_custom_metric() before "
            "loading a saved panel that references them."
        ) from exc


def list_metrics(
    category: Optional[str] = None,
    include_custom: bool = True
) -> List[str]:
    """
    List metrics in the global registry.

    Parameters
    ----------
    category : str, optional
        Filter by category.
    include_custom : bool, default True
        Whether to include custom metrics.

    Returns
    -------
    list of str
        Metric names.
    """
    return _registry.list_metrics(category, include_custom)


def list_categories() -> List[str]:
    """List metric categories in the global registry."""
    return _registry.list_categories()


def get_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _registry


def describe_metric(name: str) -> str:
    """
    Get a detailed description of a metric.

    Parameters
    ----------
    name : str
        Metric name.

    Returns
    -------
    str
        Formatted description.
    """
    return _registry.describe(name)
