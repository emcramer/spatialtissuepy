"""
Plot configuration, themes, and style settings.

This module provides consistent styling for publication-quality figures,
color palettes for cell types and categorical data, and utilities for
figure export.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np

# Lazy imports for matplotlib
_MPL_AVAILABLE = None


def _check_matplotlib():
    """Check if matplotlib is available."""
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
    return True


# -----------------------------------------------------------------------------
# Default Color Palettes
# -----------------------------------------------------------------------------

# Cell type colors - designed for distinguishability and colorblind-friendliness
CELL_TYPE_COLORS = {
    # Immune cells
    'T_cell': '#1f77b4',       # Blue
    'CD4_T_cell': '#1f77b4',
    'CD8_T_cell': '#aec7e8',   # Light blue
    'Treg': '#17becf',         # Cyan
    'B_cell': '#2ca02c',       # Green
    'Macrophage': '#ff7f0e',   # Orange
    'M1_Macrophage': '#ff7f0e',
    'M2_Macrophage': '#ffbb78',
    'Dendritic': '#9467bd',    # Purple
    'NK_cell': '#8c564b',      # Brown
    'Neutrophil': '#e377c2',   # Pink
    'Monocyte': '#7f7f7f',     # Gray
    
    # Tumor cells
    'Tumor': '#d62728',        # Red
    'Cancer': '#d62728',
    'Tumor_proliferating': '#ff9896',
    'Tumor_hypoxic': '#8b0000',
    
    # Stromal cells
    'Fibroblast': '#bcbd22',   # Yellow-green
    'CAF': '#bcbd22',
    'Endothelial': '#17becf',  # Cyan
    'Epithelial': '#98df8a',   # Light green
    
    # Other
    'Unknown': '#c7c7c7',
    'Other': '#c7c7c7',
}

# Categorical palettes
CATEGORICAL_PALETTES = {
    'default': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'colorblind': [
        '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
        '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'
    ],
    'pastel': [
        '#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6',
        '#ffff99', '#b15928', '#e31a1c', '#33a02c', '#1f78b4'
    ],
    'dark': [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
        '#e6ab02', '#a6761d', '#666666', '#e41a1c', '#377eb8'
    ],
}

# Sequential colormaps for continuous data
SEQUENTIAL_CMAPS = {
    'density': 'viridis',
    'expression': 'magma',
    'distance': 'plasma',
    'enrichment': 'YlOrRd',
    'count': 'Blues',
}

# Diverging colormaps for data with meaningful center
DIVERGING_CMAPS = {
    'correlation': 'RdBu_r',
    'log_fold_change': 'RdBu_r',
    'residual': 'coolwarm',
    'difference': 'PiYG',
}


# -----------------------------------------------------------------------------
# Plot Configuration
# -----------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """
    Configuration class for plot styling.
    
    Attributes
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    dpi : int
        Resolution for raster outputs.
    font_family : str
        Font family for text.
    font_size : int
        Base font size.
    title_size : int
        Font size for titles.
    label_size : int
        Font size for axis labels.
    tick_size : int
        Font size for tick labels.
    legend_size : int
        Font size for legends.
    line_width : float
        Default line width.
    marker_size : float
        Default marker size.
    alpha : float
        Default transparency.
    """
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    font_family: str = 'sans-serif'
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9
    line_width: float = 1.5
    marker_size: float = 20
    alpha: float = 0.7
    
    # Color settings
    cell_type_colors: Dict[str, str] = field(default_factory=lambda: CELL_TYPE_COLORS.copy())
    categorical_palette: str = 'default'
    sequential_cmap: str = 'viridis'
    diverging_cmap: str = 'RdBu_r'
    
    # Grid and spines
    show_grid: bool = False
    grid_alpha: float = 0.3
    despine: bool = True
    
    def apply(self):
        """Apply this configuration to matplotlib."""
        _check_matplotlib()
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            'lines.linewidth': self.line_width,
            'lines.markersize': np.sqrt(self.marker_size),
            'axes.grid': self.show_grid,
            'grid.alpha': self.grid_alpha,
        })


# Global configuration instance
_current_config = PlotConfig()


def get_config() -> PlotConfig:
    """Get current plot configuration."""
    return _current_config


def set_config(config: PlotConfig):
    """Set global plot configuration."""
    global _current_config
    _current_config = config
    config.apply()


# -----------------------------------------------------------------------------
# Style Presets
# -----------------------------------------------------------------------------

def set_publication_style(
    journal: str = 'default',
    column_width: str = 'single'
) -> PlotConfig:
    """
    Set matplotlib style for publication-quality figures.
    
    Parameters
    ----------
    journal : str, default 'default'
        Target journal style: 'default', 'nature', 'science', 'cell'.
    column_width : str, default 'single'
        Column width: 'single', 'double', 'full'.
        
    Returns
    -------
    PlotConfig
        Applied configuration.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    # Journal-specific widths (inches)
    widths = {
        'default': {'single': 3.5, 'double': 7.0, 'full': 7.5},
        'nature': {'single': 3.5, 'double': 7.0, 'full': 7.5},
        'science': {'single': 3.4, 'double': 7.0, 'full': 7.0},
        'cell': {'single': 3.3, 'double': 6.9, 'full': 7.0},
    }
    
    width = widths.get(journal, widths['default']).get(column_width, 3.5)
    height = width * 0.75  # Default aspect ratio
    
    config = PlotConfig(
        figsize=(width, height),
        dpi=300,
        font_family='sans-serif',
        font_size=8,
        title_size=9,
        label_size=8,
        tick_size=7,
        legend_size=7,
        line_width=1.0,
        marker_size=15,
        alpha=0.8,
        show_grid=False,
        despine=True,
    )
    
    set_config(config)
    
    # Additional matplotlib settings for publication
    plt.rcParams.update({
        'pdf.fonttype': 42,  # TrueType fonts for PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',  # Text as text in SVG
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })
    
    return config


def set_default_style() -> PlotConfig:
    """
    Reset to default plotting style.
    
    Returns
    -------
    PlotConfig
        Applied configuration.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    plt.rcdefaults()
    
    config = PlotConfig()
    set_config(config)
    
    return config


def set_presentation_style() -> PlotConfig:
    """
    Set style for presentations (larger fonts, bolder lines).
    
    Returns
    -------
    PlotConfig
        Applied configuration.
    """
    _check_matplotlib()
    
    config = PlotConfig(
        figsize=(10, 7),
        dpi=150,
        font_size=14,
        title_size=16,
        label_size=14,
        tick_size=12,
        legend_size=12,
        line_width=2.0,
        marker_size=50,
        alpha=0.8,
        show_grid=False,
        despine=True,
    )
    
    set_config(config)
    
    return config


# -----------------------------------------------------------------------------
# Color Utilities
# -----------------------------------------------------------------------------

def get_cell_type_colors(
    cell_types: Optional[List[str]] = None,
    palette: str = 'default'
) -> Dict[str, str]:
    """
    Get color mapping for cell types.
    
    Parameters
    ----------
    cell_types : list of str, optional
        Cell types to get colors for. If None, returns all known colors.
    palette : str, default 'default'
        Palette to use for unknown cell types.
        
    Returns
    -------
    dict
        Mapping from cell type to hex color.
    """
    colors = _current_config.cell_type_colors.copy()
    
    if cell_types is None:
        return colors
    
    # Add colors for unknown types
    palette_colors = CATEGORICAL_PALETTES.get(palette, CATEGORICAL_PALETTES['default'])
    unknown_idx = 0
    
    result = {}
    for ct in cell_types:
        if ct in colors:
            result[ct] = colors[ct]
        else:
            result[ct] = palette_colors[unknown_idx % len(palette_colors)]
            unknown_idx += 1
    
    return result


def get_categorical_palette(
    n_colors: int,
    palette: str = 'default'
) -> List[str]:
    """
    Get a categorical color palette.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed.
    palette : str, default 'default'
        Palette name: 'default', 'colorblind', 'pastel', 'dark'.
        
    Returns
    -------
    list of str
        List of hex colors.
    """
    colors = CATEGORICAL_PALETTES.get(palette, CATEGORICAL_PALETTES['default'])
    
    if n_colors <= len(colors):
        return colors[:n_colors]
    
    # Extend by cycling
    extended = []
    for i in range(n_colors):
        extended.append(colors[i % len(colors)])
    
    return extended


def get_sequential_cmap(
    name: str = 'density'
) -> str:
    """
    Get a sequential colormap name.
    
    Parameters
    ----------
    name : str, default 'density'
        Type of data: 'density', 'expression', 'distance', 'enrichment', 'count'.
        
    Returns
    -------
    str
        Matplotlib colormap name.
    """
    return SEQUENTIAL_CMAPS.get(name, 'viridis')


def get_diverging_cmap(
    name: str = 'correlation'
) -> str:
    """
    Get a diverging colormap name.
    
    Parameters
    ----------
    name : str, default 'correlation'
        Type of data: 'correlation', 'log_fold_change', 'residual', 'difference'.
        
    Returns
    -------
    str
        Matplotlib colormap name.
    """
    return DIVERGING_CMAPS.get(name, 'RdBu_r')


# -----------------------------------------------------------------------------
# Figure Utilities
# -----------------------------------------------------------------------------

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple['plt.Figure', Union['plt.Axes', np.ndarray]]:
    """
    Create a figure with consistent styling.
    
    Parameters
    ----------
    nrows : int, default 1
        Number of rows.
    ncols : int, default 1
        Number of columns.
    figsize : tuple, optional
        Figure size. If None, uses config default scaled by grid.
    **kwargs
        Additional arguments to plt.subplots().
        
    Returns
    -------
    fig : plt.Figure
        Figure object.
    axes : plt.Axes or array of plt.Axes
        Axes object(s).
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    if figsize is None:
        base = _current_config.figsize
        figsize = (base[0] * ncols, base[1] * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    return fig, axes


def save_figure(
    fig: 'plt.Figure',
    filename: str,
    formats: List[str] = None,
    dpi: Optional[int] = None,
    transparent: bool = False,
    bbox_inches: str = 'tight',
    **kwargs
):
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    filename : str
        Base filename (without extension).
    formats : list of str, optional
        Output formats. Default: ['pdf', 'png'].
    dpi : int, optional
        Resolution for raster formats. Uses config default if None.
    transparent : bool, default False
        Transparent background.
    bbox_inches : str, default 'tight'
        Bounding box setting.
    **kwargs
        Additional arguments to fig.savefig().
    """
    _check_matplotlib()
    
    if formats is None:
        formats = ['pdf', 'png']
    
    if dpi is None:
        dpi = _current_config.dpi
    
    for fmt in formats:
        output_path = f"{filename}.{fmt}"
        fig.savefig(
            output_path,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            **kwargs
        )


def despine(
    ax: 'plt.Axes',
    left: bool = False,
    bottom: bool = False,
    right: bool = True,
    top: bool = True
):
    """
    Remove spines from axes.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to modify.
    left, bottom, right, top : bool
        Which spines to remove.
    """
    for spine, remove in [('left', left), ('bottom', bottom), 
                          ('right', right), ('top', top)]:
        ax.spines[spine].set_visible(not remove)


def add_scalebar(
    ax: 'plt.Axes',
    length: float,
    unit: str = 'µm',
    location: str = 'lower right',
    color: str = 'black',
    fontsize: Optional[int] = None
):
    """
    Add a scale bar to spatial plots.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to add scale bar to.
    length : float
        Length of scale bar in data units.
    unit : str, default 'µm'
        Unit label.
    location : str, default 'lower right'
        Location: 'lower right', 'lower left', 'upper right', 'upper left'.
    color : str, default 'black'
        Scale bar color.
    fontsize : int, optional
        Font size for label.
    """
    _check_matplotlib()
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
    
    if fontsize is None:
        fontsize = _current_config.tick_size
    
    # Get axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Position based on location
    if 'lower' in location:
        y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    else:
        y = ylim[1] - 0.1 * (ylim[1] - ylim[0])
    
    if 'right' in location:
        x = xlim[1] - 0.05 * (xlim[0] - xlim[0]) - length
    else:
        x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    
    # Draw scale bar
    bar_height = 0.01 * (ylim[1] - ylim[0])
    ax.add_patch(Rectangle(
        (x, y), length, bar_height,
        facecolor=color, edgecolor=color
    ))
    
    # Add label
    ax.text(
        x + length / 2, y - bar_height * 2,
        f'{length} {unit}',
        ha='center', va='top',
        fontsize=fontsize, color=color
    )


def get_axes(
    ax: Optional['plt.Axes'] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> 'plt.Axes':
    """
    Get or create axes for plotting.
    
    Parameters
    ----------
    ax : plt.Axes, optional
        Existing axes. If None, creates new figure.
    figsize : tuple, optional
        Figure size if creating new figure.
        
    Returns
    -------
    plt.Axes
        Axes object.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    if ax is None:
        if figsize is None:
            figsize = _current_config.figsize
        fig, ax = plt.subplots(figsize=figsize)
    
    return ax
