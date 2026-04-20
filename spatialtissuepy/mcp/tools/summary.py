"""
Summary tools for MCP server.

Feature extraction tools for ML pipelines.

Tools (8 total):
- summary_create_panel: Create metric panel
- summary_add_metric: Add metric to panel
- summary_list_available_metrics: List registered metrics
- summary_compute: Compute single-sample summary
- summary_to_dict: Get as dictionary
- summary_to_array: Get as feature vector
- summary_multi_sample: Multi-sample features
- summary_multi_sample_to_dataframe: Get as DataFrame
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


class PanelInfo(BaseModel):
    """Information about a statistics panel."""

    session_id: str
    panel_key: str
    name: str
    n_metrics: int
    metrics: List[str]


class MetricInfo(BaseModel):
    """Information about a registered metric."""

    name: str
    category: str
    description: str
    parameters: Dict[str, str]


class MetricsList(BaseModel):
    """List of available metrics."""

    metrics: List[MetricInfo]
    categories: List[str]
    total: int


class SummaryResult(BaseModel):
    """Result of computing a summary."""

    session_id: str
    data_key: str
    panel_key: str
    n_features: int
    feature_names: List[str]
    features: Dict[str, float]


class MultiSampleResult(BaseModel):
    """Result of multi-sample summary."""

    session_id: str
    panel_key: str
    n_samples: int
    n_features: int
    sample_ids: List[str]
    feature_names: List[str]


def register_tools(mcp: "FastMCP") -> None:
    """Register summary tools with the MCP server."""

    @mcp.tool()
    def summary_create_panel(
        session_id: str,
        panel_key: str = "custom_panel",
        name: str = "Custom Panel",
        preset: Optional[str] = None,
    ) -> PanelInfo:
        """
        Create a statistics panel for feature extraction.

        Panels define which metrics to compute. Use presets or build custom.

        Parameters
        ----------
        session_id : str
            Session to store the panel in.
        panel_key : str
            Key to store the panel.
        name : str
            Human-readable panel name.
        preset : str, optional
            Use a preset: "basic", "spatial", "comprehensive".

        Returns
        -------
        PanelInfo
            Information about the created panel.
        """
        from spatialtissuepy.summary import StatisticsPanel, load_panel
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        if preset:
            panel = load_panel(preset)
            panel.name = name
        else:
            panel = StatisticsPanel(name=name)

        session_mgr.store_panel(session_id, panel_key, panel)

        return PanelInfo(
            session_id=session_id,
            panel_key=panel_key,
            name=name,
            n_metrics=len(panel.metrics) if hasattr(panel, "metrics") else 0,
            metrics=[m.name for m in panel.metrics] if hasattr(panel, "metrics") else [],
        )

    @mcp.tool()
    def summary_add_metric(
        session_id: str,
        metric_name: str,
        panel_key: str = "custom_panel",
        metric_params: Optional[Dict[str, Any]] = None,
    ) -> PanelInfo:
        """
        Add a metric to an existing panel.

        Parameters
        ----------
        session_id : str
            Session containing the panel.
        metric_name : str
            Name of the metric to add (use summary_list_available_metrics).
        panel_key : str
            Key of the panel.
        metric_params : dict, optional
            Additional parameters for the metric (e.g., {"radius": 50, "type_a": "Tumor"}).

        Returns
        -------
        PanelInfo
            Updated panel information.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        panel = session_mgr.load_panel(session_id, panel_key)

        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        params = metric_params or {}
        panel.add(metric_name, **params)
        session_mgr.store_panel(session_id, panel_key, panel)

        return PanelInfo(
            session_id=session_id,
            panel_key=panel_key,
            name=panel.name,
            n_metrics=len(panel.metrics),
            metrics=[m.name for m in panel.metrics],
        )

    @mcp.tool()
    def summary_list_available_metrics(
        category: Optional[str] = None,
    ) -> MetricsList:
        """
        List all available registered metrics.

        Parameters
        ----------
        category : str, optional
            Filter by category.

        Returns
        -------
        MetricsList
            Available metrics and categories.
        """
        from spatialtissuepy.summary import list_metrics, list_categories, get_metric

        categories = list_categories()
        all_metrics = list_metrics(category=category)

        metrics_info = []
        for name in all_metrics:
            try:
                info = get_metric(name)
                metrics_info.append(MetricInfo(
                    name=name,
                    category=info.category if hasattr(info, "category") else "unknown",
                    description=info.description if hasattr(info, "description") else "",
                    parameters={k: str(v) for k, v in (info.parameters if hasattr(info, "parameters") else {}).items()},
                ))
            except Exception:
                metrics_info.append(MetricInfo(
                    name=name,
                    category="unknown",
                    description="",
                    parameters={},
                ))

        return MetricsList(
            metrics=metrics_info,
            categories=categories,
            total=len(metrics_info),
        )

    @mcp.tool()
    def summary_compute(
        session_id: str,
        data_key: str = "primary",
        panel_key: str = "custom_panel",
    ) -> SummaryResult:
        """
        Compute summary features for a dataset using a panel.

        Parameters
        ----------
        session_id : str
            Session containing the data and panel.
        data_key : str
            Key of the spatial data.
        panel_key : str
            Key of the statistics panel.

        Returns
        -------
        SummaryResult
            Computed features as dictionary.
        """
        from spatialtissuepy.summary import SpatialSummary
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)
        panel = session_mgr.load_panel(session_id, panel_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")
        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        summary = SpatialSummary(data, panel)
        features = summary.to_dict()

        return SummaryResult(
            session_id=session_id,
            data_key=data_key,
            panel_key=panel_key,
            n_features=len(features),
            feature_names=list(features.keys()),
            features={k: float(v) if isinstance(v, (int, float)) else v for k, v in features.items()},
        )

    @mcp.tool()
    def summary_to_dict(
        session_id: str,
        data_key: str = "primary",
        panel_key: str = "custom_panel",
    ) -> Dict[str, Any]:
        """
        Compute summary and return as dictionary.

        Parameters
        ----------
        session_id : str
            Session containing the data and panel.
        data_key : str
            Key of the spatial data.
        panel_key : str
            Key of the statistics panel.

        Returns
        -------
        dict
            Feature name to value mapping.
        """
        # Call the underlying library directly rather than the decorated
        # summary_compute() tool -- FastMCP wraps @mcp.tool() functions in
        # a FunctionTool object that is not directly callable.
        from spatialtissuepy.summary import SpatialSummary
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)
        panel = session_mgr.load_panel(session_id, panel_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")
        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        summary = SpatialSummary(data, panel)
        features = summary.to_dict()
        return {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in features.items()
        }

    @mcp.tool()
    def summary_to_array(
        session_id: str,
        data_key: str = "primary",
        panel_key: str = "custom_panel",
    ) -> Dict[str, Any]:
        """
        Compute summary and return as feature vector.

        Parameters
        ----------
        session_id : str
            Session containing the data and panel.
        data_key : str
            Key of the spatial data.
        panel_key : str
            Key of the statistics panel.

        Returns
        -------
        dict
            Feature array and names.
        """
        from spatialtissuepy.summary import SpatialSummary
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)
        panel = session_mgr.load_panel(session_id, panel_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")
        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        summary = SpatialSummary(data, panel)
        array = summary.to_array()
        names = summary.feature_names

        return {
            "values": array.tolist(),
            "feature_names": names,
            "n_features": len(array),
        }

    @mcp.tool()
    def summary_multi_sample(
        session_id: str,
        data_keys: List[str],
        panel_key: str = "custom_panel",
        sample_ids: Optional[List[str]] = None,
    ) -> MultiSampleResult:
        """
        Compute features for multiple samples.

        Parameters
        ----------
        session_id : str
            Session containing the data and panel.
        data_keys : list of str
            Keys of spatial data objects.
        panel_key : str
            Key of the statistics panel.
        sample_ids : list of str, optional
            Sample identifiers. Default: data_keys.

        Returns
        -------
        MultiSampleResult
            Summary of computed features.
        """
        from spatialtissuepy.summary import MultiSampleSummary
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        panel = session_mgr.load_panel(session_id, panel_key)

        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        samples = []
        for key in data_keys:
            data = session_mgr.load_data(session_id, key)
            if data is None:
                raise ValueError(f"No data found with key '{key}'")
            samples.append(data)

        if sample_ids is None:
            sample_ids = data_keys

        multi = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        feature_names = multi.feature_names

        return MultiSampleResult(
            session_id=session_id,
            panel_key=panel_key,
            n_samples=len(samples),
            n_features=len(feature_names),
            sample_ids=sample_ids,
            feature_names=feature_names,
        )

    @mcp.tool()
    def summary_multi_sample_to_dataframe(
        session_id: str,
        data_keys: List[str],
        panel_key: str = "custom_panel",
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute multi-sample features as DataFrame.

        Returns a tabular format ready for ML pipelines.

        Parameters
        ----------
        session_id : str
            Session containing the data and panel.
        data_keys : list of str
            Keys of spatial data objects.
        panel_key : str
            Key of the statistics panel.
        sample_ids : list of str, optional
            Sample identifiers.

        Returns
        -------
        dict
            DataFrame-like structure with samples as rows.
        """
        from spatialtissuepy.summary import MultiSampleSummary
        from ..server import get_session_manager, get_serializer

        session_mgr = get_session_manager()
        serializer = get_serializer()
        panel = session_mgr.load_panel(session_id, panel_key)

        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        samples = []
        for key in data_keys:
            data = session_mgr.load_data(session_id, key)
            if data is None:
                raise ValueError(f"No data found with key '{key}'")
            samples.append(data)

        if sample_ids is None:
            sample_ids = data_keys

        multi = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        df = multi.to_dataframe()

        return serializer.dataframe_to_json(df)
