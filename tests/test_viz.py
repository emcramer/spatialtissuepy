"""
Tests for spatialtissuepy.viz module.

Tests visualization functions for creating publication-quality plots.
Focus on function signatures, parameter validation, and data handling
rather than visual output quality.
"""

import pytest
import numpy as np
import pandas as pd

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from spatialtissuepy import SpatialTissueData

if HAS_MATPLOTLIB:
    from spatialtissuepy.viz import (
        # Config
        set_publication_style,
        set_default_style,
        get_cell_type_colors,
        get_categorical_palette,
        save_figure,
        PlotConfig,
        # Spatial plots
        plot_spatial_scatter,
        plot_cell_types,
        plot_marker_expression,
        plot_density_map,
    )


pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed"
)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestVizConfiguration:
    """Tests for visualization configuration."""
    
    def test_set_publication_style(self):
        """Test setting publication style."""
        config = set_publication_style()
        assert isinstance(config, PlotConfig)
        assert config.dpi == 300
    
    def test_set_default_style(self):
        """Test setting default style."""
        config = set_default_style()
        assert isinstance(config, PlotConfig)
    
    def test_get_cell_type_colors(self):
        """Test getting cell type color mapping."""
        cell_types = ['Tumor', 'T_cell']
        colors = get_cell_type_colors(cell_types)
        
        assert isinstance(colors, dict)
        assert 'Tumor' in colors
        assert 'T_cell' in colors
    
    def test_get_categorical_palette(self):
        """Test getting categorical color palette."""
        palette = get_categorical_palette(n_colors=5)
        assert len(palette) == 5


# =============================================================================
# Spatial Plot Tests
# =============================================================================

class TestSpatialPlots:
    """Tests for spatial plotting functions."""
    
    def test_plot_spatial_scatter_basic(self, small_tissue):
        """Test basic spatial scatter plot."""
        ax = plot_spatial_scatter(small_tissue)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)
    
    def test_plot_spatial_scatter_color_by_marker(self, tissue_with_markers):
        """Test coloring by marker."""
        marker = tissue_with_markers.marker_names[0]
        ax = plot_spatial_scatter(tissue_with_markers, marker=marker)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_cell_types(self, small_tissue):
        """Test plotting cell types (faceted)."""
        # Returns a Figure
        fig = plot_cell_types(small_tissue, ncols=2)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    
    def test_plot_marker_expression(self, tissue_with_markers):
        """Test plotting marker expression (faceted)."""
        markers = tissue_with_markers.marker_names[:2]
        fig = plot_marker_expression(tissue_with_markers, markers=markers)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    
    def test_plot_density_map(self, small_tissue):
        """Test plotting density map."""
        ax = plot_density_map(small_tissue)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestVizEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_plot_with_invalid_marker(self, small_tissue):
        """Test error with nonexistent marker."""
        with pytest.raises(ValueError):
            plot_spatial_scatter(small_tissue, marker='nonexistent')
    
    def test_plot_with_wrong_data_type(self):
        """Test plotting with wrong data type."""
        with pytest.raises(AttributeError):
            plot_spatial_scatter("not a data object")

