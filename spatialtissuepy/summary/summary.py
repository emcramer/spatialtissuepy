"""
Spatial summary computation for single and multiple samples.

This module provides classes for computing spatial statistics summaries
that can be used as feature vectors for downstream analysis.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Dict, Iterator, List, Optional, 
    Tuple, Union
)
import numpy as np
import pandas as pd
from pathlib import Path

from .panel import StatisticsPanel, load_panel

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


class SpatialSummary:
    """
    Compute spatial statistics summary for a single sample.
    
    The summary is a 1D vector where each element is a different
    spatial statistic, providing a compact description of the
    tissue's spatial organization.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input spatial data.
    panel : StatisticsPanel or str
        Panel of statistics to compute. Can be a StatisticsPanel
        object or name of a predefined panel.
    
    Attributes
    ----------
    results : dict
        Raw results as metric_name -> value mapping.
    column_names : list of str
        Names of all columns in order.
    
    Examples
    --------
    >>> from spatialtissuepy.summary import SpatialSummary, load_panel
    >>> 
    >>> # Using predefined panel
    >>> summary = SpatialSummary(data, panel='basic')
    >>> vector = summary.to_array()
    >>> 
    >>> # Using custom panel
    >>> panel = StatisticsPanel()
    >>> panel.add('cell_counts')
    >>> panel.add('ripleys_k', radii=[50, 100])
    >>> summary = SpatialSummary(data, panel)
    """
    
    def __init__(
        self,
        data: 'SpatialTissueData',
        panel: Union[StatisticsPanel, str],
    ):
        self.data = data
        
        # Resolve panel
        if isinstance(panel, str):
            self.panel = load_panel(panel)
        else:
            self.panel = panel
        
        # Compute summary
        self._results: Dict[str, float] = {}
        self._column_names: List[str] = []
        self._compute()
    
    def _compute(self) -> None:
        """Compute all metrics in the panel."""
        self._results = self.panel.compute(self.data)
        self._column_names = list(self._results.keys())
    
    @property
    def results(self) -> Dict[str, float]:
        """Raw results as dictionary."""
        return self._results.copy()
    
    @property
    def column_names(self) -> List[str]:
        """Ordered list of column names."""
        return self._column_names.copy()
    
    @property
    def n_features(self) -> int:
        """Number of features in the summary vector."""
        return len(self._column_names)
    
    def to_array(self) -> np.ndarray:
        """
        Convert summary to 1D numpy array.
        
        Returns
        -------
        np.ndarray
            Shape (n_features,) array of statistic values.
        """
        return np.array([self._results[col] for col in self._column_names])
    
    def to_series(self, name: Optional[str] = None) -> pd.Series:
        """
        Convert summary to pandas Series.
        
        Parameters
        ----------
        name : str, optional
            Series name (e.g., sample ID).
        
        Returns
        -------
        pd.Series
            Series with metric names as index.
        """
        return pd.Series(self._results, name=name)
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert summary to dictionary.
        
        Returns
        -------
        dict
            Metric names to values.
        """
        return self._results.copy()
    
    def get(self, metric_name: str, default: float = np.nan) -> float:
        """
        Get a specific metric value.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric.
        default : float
            Value to return if metric not found.
        
        Returns
        -------
        float
            Metric value.
        """
        return self._results.get(metric_name, default)
    
    def __getitem__(self, key: str) -> float:
        return self._results[key]
    
    def __len__(self) -> int:
        return len(self._results)
    
    def __repr__(self) -> str:
        return f"SpatialSummary({self.n_features} features)"
    
    def __str__(self) -> str:
        lines = [
            f"SpatialSummary ({self.n_features} features)",
            f"  Panel: {self.panel.name}",
            "  First 10 metrics:",
        ]
        
        for col in self._column_names[:10]:
            val = self._results[col]
            if np.isnan(val):
                lines.append(f"    {col}: NaN")
            else:
                lines.append(f"    {col}: {val:.4g}")
        
        if len(self._column_names) > 10:
            lines.append(f"    ... and {len(self._column_names) - 10} more")
        
        return '\n'.join(lines)


class MultiSampleSummary:
    """
    Compute spatial statistics summaries for multiple samples.
    
    Returns a DataFrame where each row is a sample and each column
    is a spatial statistic.
    
    Parameters
    ----------
    samples : list of SpatialTissueData
        List of samples to summarize.
    panel : StatisticsPanel or str
        Panel of statistics to compute.
    sample_ids : list of str, optional
        Sample identifiers. If None, uses 0-indexed integers.
    n_jobs : int, default 1
        Number of parallel jobs. -1 uses all CPUs.
    show_progress : bool, default True
        Show progress bar.
    
    Examples
    --------
    >>> from spatialtissuepy.summary import MultiSampleSummary
    >>> 
    >>> # From list of samples
    >>> samples = [data1, data2, data3]
    >>> multi = MultiSampleSummary(samples, panel='basic')
    >>> df = multi.to_dataframe()
    >>> 
    >>> # From multi-sample data
    >>> multi = MultiSampleSummary.from_multisample(data, panel='basic')
    >>> df = multi.to_dataframe()
    >>> 
    >>> # Export for ML
    >>> df.to_csv('spatial_features.csv')
    """
    
    def __init__(
        self,
        samples: List['SpatialTissueData'],
        panel: Union[StatisticsPanel, str],
        sample_ids: Optional[List[str]] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ):
        self.samples = samples
        
        # Resolve panel
        if isinstance(panel, str):
            self.panel = load_panel(panel)
        else:
            self.panel = panel
        
        # Sample IDs
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(samples))]
        
        if len(sample_ids) != len(samples):
            raise ValueError(
                f"Length of sample_ids ({len(sample_ids)}) must match "
                f"number of samples ({len(samples)})"
            )
        
        self.sample_ids = list(sample_ids)
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        
        # Compute summaries
        self._summaries: List[SpatialSummary] = []
        self._df: Optional[pd.DataFrame] = None
        self._compute()
    
    @classmethod
    def from_multisample(
        cls,
        data: 'SpatialTissueData',
        panel: Union[StatisticsPanel, str],
        **kwargs
    ) -> 'MultiSampleSummary':
        """
        Create from a multi-sample SpatialTissueData object.
        
        Parameters
        ----------
        data : SpatialTissueData
            Multi-sample data (must have sample_ids).
        panel : StatisticsPanel or str
            Panel to compute.
        **kwargs
            Additional arguments for MultiSampleSummary.
        
        Returns
        -------
        MultiSampleSummary
            Summary for all samples.
        """
        if not data.is_multisample:
            raise ValueError("Data must be multi-sample (have sample_ids)")
        
        samples = []
        sample_ids = []
        
        for sample_id, sample_data in data.iter_samples():
            samples.append(sample_data)
            sample_ids.append(sample_id)
        
        return cls(samples, panel, sample_ids=sample_ids, **kwargs)
    
    def _compute(self) -> None:
        """Compute summaries for all samples."""
        if self.n_jobs == 1:
            self._compute_sequential()
        else:
            self._compute_parallel()
        
        self._build_dataframe()
    
    def _compute_sequential(self) -> None:
        """Compute summaries sequentially."""
        samples_iter = self.samples
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                samples_iter = tqdm(
                    self.samples, 
                    desc="Computing summaries",
                    total=len(self.samples)
                )
            except ImportError:
                pass
        
        for sample in samples_iter:
            summary = SpatialSummary(sample, self.panel)
            self._summaries.append(summary)
    
    def _compute_parallel(self) -> None:
        """Compute summaries in parallel."""
        try:
            from joblib import Parallel, delayed
        except ImportError:
            # Fall back to sequential
            self._compute_sequential()
            return
        
        n_jobs = self.n_jobs
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1
        
        def compute_one(sample):
            return SpatialSummary(sample, self.panel)
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                self._summaries = Parallel(n_jobs=n_jobs)(
                    delayed(compute_one)(s) 
                    for s in tqdm(self.samples, desc="Computing summaries")
                )
            except ImportError:
                self._summaries = Parallel(n_jobs=n_jobs)(
                    delayed(compute_one)(s) for s in self.samples
                )
        else:
            self._summaries = Parallel(n_jobs=n_jobs)(
                delayed(compute_one)(s) for s in self.samples
            )
    
    def _build_dataframe(self) -> None:
        """Build DataFrame from summaries."""
        if not self._summaries:
            self._df = pd.DataFrame()
            return
        
        # Get all column names (union across samples)
        all_columns = set()
        for summary in self._summaries:
            all_columns.update(summary.column_names)
        
        # Sort columns for consistency
        columns = sorted(all_columns)
        
        # Build rows
        rows = []
        for summary in self._summaries:
            row = {col: summary.get(col, np.nan) for col in columns}
            rows.append(row)
        
        self._df = pd.DataFrame(rows, index=self.sample_ids)
        self._df.index.name = 'sample_id'
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.samples)
    
    @property
    def n_features(self) -> int:
        """Number of features (columns)."""
        return len(self._df.columns) if self._df is not None else 0
    
    @property
    def column_names(self) -> List[str]:
        """List of column names."""
        return list(self._df.columns) if self._df is not None else []
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get summary as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Rows are samples, columns are metrics.
        """
        return self._df.copy()
    
    def to_array(self) -> np.ndarray:
        """
        Get summary as 2D numpy array.
        
        Returns
        -------
        np.ndarray
            Shape (n_samples, n_features).
        """
        return self._df.values.copy()
    
    def to_csv(
        self,
        filepath: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Export to CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Output filepath.
        **kwargs
            Additional arguments for pd.DataFrame.to_csv.
        """
        self._df.to_csv(filepath, **kwargs)
    
    def to_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: str = 'spatial_summary',
        **kwargs
    ) -> None:
        """
        Export to Excel file.
        
        Parameters
        ----------
        filepath : str or Path
            Output filepath.
        sheet_name : str
            Excel sheet name.
        **kwargs
            Additional arguments for pd.DataFrame.to_excel.
        """
        self._df.to_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    def get_sample(self, sample_id: str) -> SpatialSummary:
        """
        Get summary for a specific sample.
        
        Parameters
        ----------
        sample_id : str
            Sample identifier.
        
        Returns
        -------
        SpatialSummary
            Summary for the sample.
        """
        try:
            idx = self.sample_ids.index(sample_id)
            return self._summaries[idx]
        except ValueError:
            raise KeyError(f"Sample '{sample_id}' not found")
    
    def get_metric(self, metric_name: str) -> pd.Series:
        """
        Get values of a specific metric across all samples.
        
        Parameters
        ----------
        metric_name : str
            Metric name.
        
        Returns
        -------
        pd.Series
            Values indexed by sample_id.
        """
        if metric_name not in self._df.columns:
            raise KeyError(f"Metric '{metric_name}' not in summary")
        return self._df[metric_name]
    
    def describe(self) -> pd.DataFrame:
        """
        Compute summary statistics for each metric.
        
        Returns
        -------
        pd.DataFrame
            Standard pandas describe output.
        """
        return self._df.describe()
    
    def dropna(
        self,
        axis: int = 1,
        how: str = 'any',
        thresh: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Return DataFrame with NaN values handled.
        
        Parameters
        ----------
        axis : int
            0 = drop rows, 1 = drop columns.
        how : str
            'any' or 'all'.
        thresh : int, optional
            Require this many non-NA values.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with NA handled.
        """
        return self._df.dropna(axis=axis, how=how, thresh=thresh)
    
    def __getitem__(self, key: str) -> pd.Series:
        """Get a metric column."""
        return self._df[key]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __iter__(self) -> Iterator[Tuple[str, SpatialSummary]]:
        """Iterate over (sample_id, summary) pairs."""
        for sample_id, summary in zip(self.sample_ids, self._summaries):
            yield sample_id, summary
    
    def __repr__(self) -> str:
        return f"MultiSampleSummary({self.n_samples} samples, {self.n_features} features)"
    
    def __str__(self) -> str:
        lines = [
            f"MultiSampleSummary",
            f"  Samples: {self.n_samples}",
            f"  Features: {self.n_features}",
            f"  Panel: {self.panel.name}",
            "",
            "  Sample IDs:",
        ]
        
        for sid in self.sample_ids[:5]:
            lines.append(f"    - {sid}")
        
        if len(self.sample_ids) > 5:
            lines.append(f"    ... and {len(self.sample_ids) - 5} more")
        
        return '\n'.join(lines)


def compute_summary(
    data: 'SpatialTissueData',
    panel: Union[StatisticsPanel, str] = 'basic',
) -> pd.Series:
    """
    Convenience function to compute summary for a single sample.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    panel : StatisticsPanel or str, default 'basic'
        Panel to compute.
    
    Returns
    -------
    pd.Series
        Summary as Series with metric names as index.
    """
    summary = SpatialSummary(data, panel)
    return summary.to_series()


def compute_multi_summary(
    samples: List['SpatialTissueData'],
    panel: Union[StatisticsPanel, str] = 'basic',
    sample_ids: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to compute summaries for multiple samples.
    
    Parameters
    ----------
    samples : list of SpatialTissueData
        Input samples.
    panel : StatisticsPanel or str, default 'basic'
        Panel to compute.
    sample_ids : list of str, optional
        Sample identifiers.
    **kwargs
        Additional arguments for MultiSampleSummary.
    
    Returns
    -------
    pd.DataFrame
        Summary DataFrame.
    """
    multi = MultiSampleSummary(
        samples, panel, sample_ids=sample_ids, **kwargs
    )
    return multi.to_dataframe()
