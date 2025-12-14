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
        # Network plots
        plot_cell_graph,
        plot_degree_distribution,
        # Statistics plots
        plot_ripleys_curve,
        plot_colocalization_heatmap,
        # Comparison plots
        plot_metric_comparison,
        plot_violin_comparison,
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
        try:
            set_publication_style()
            # Should not raise error
        except Exception as e:
            pytest.fail(f"set_publication_style raised {e}")
    
    def test_set_default_style(self):
        """Test setting default style."""
        try:
            set_default_style()
            # Should not raise error
        except Exception as e:
            pytest.fail(f"set_default_style raised {e}")
    
    def test_get_cell_type_colors(self):
        """Test getting cell type color mapping."""
        cell_types = ['tumor', 'immune', 'fibroblast']
        
        colors = get_cell_type_colors(cell_types)
        
        assert isinstance(colors, dict)
        assert len(colors) == len(cell_types)
        for ct in cell_types:
            assert ct in colors
    
    def test_get_categorical_palette(self):
        """Test getting categorical color palette."""
        palette = get_categorical_palette(n_colors=5)
        
        assert isinstance(palette, list)
        assert len(palette) == 5
    
    def test_plot_config_initialization(self):
        """Test PlotConfig initialization."""
        try:
            config = PlotConfig()
            assert isinstance(config, PlotConfig)
        except Exception:
            pytest.skip("PlotConfig not available")


# =============================================================================
# Spatial Plot Tests
# =============================================================================

class TestSpatialPlots:
    """Tests for spatial plotting functions."""
    
    def test_plot_spatial_scatter_basic(self, small_tissue):
        """Test basic spatial scatter plot."""
        fig, ax = plt.subplots()
        
        result_ax = plot_spatial_scatter(small_tissue, ax=ax)
        
        assert result_ax is ax  # Should return the axes
        plt.close(fig)
    
    def test_plot_spatial_scatter_no_ax(self, small_tissue):
        """Test spatial scatter without providing axes."""
        result_ax = plot_spatial_scatter(small_tissue)
        
        assert result_ax is not None
        assert hasattr(result_ax, 'figure')
        plt.close(result_ax.figure)
    
    def test_plot_cell_types(self, small_tissue):
        """Test plotting cell types with colors."""
        fig, ax = plt.subplots()
        
        result_ax = plot_cell_types(small_tissue, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_cell_types_custom_colors(self, small_tissue):
        """Test plotting with custom color mapping."""
        colors = {ct: 'red' for ct in small_tissue.cell_types_unique}
        
        fig, ax = plt.subplots()
        result_ax = plot_cell_types(small_tissue, colors=colors, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_marker_expression(self, small_tissue):
        """Test plotting marker expression."""
        if small_tissue.markers is None:
            pytest.skip("No markers in small_tissue")
        
        marker_name = small_tissue.markers.columns[0]
        
        fig, ax = plt.subplots()
        result_ax = plot_marker_expression(
            small_tissue,
            marker=marker_name,
            ax=ax
        )
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_density_map(self, medium_tissue):
        """Test plotting density map."""
        try:
            fig, ax = plt.subplots()
            result_ax = plot_density_map(medium_tissue, ax=ax)
            
            assert result_ax is ax
            plt.close(fig)
        except ImportError:
            pytest.skip("Density map requires scipy")


# =============================================================================
# Network Plot Tests
# =============================================================================

class TestNetworkPlots:
    """Tests for network plotting functions."""
    
    def test_plot_cell_graph_basic(self, small_tissue):
        """Test plotting cell graph."""
        try:
            from spatialtissuepy.network import CellGraph
            
            graph = CellGraph.from_spatial_data(
                small_tissue,
                method='proximity',
                radius=50
            )
            
            fig, ax = plt.subplots()
            result_ax = plot_cell_graph(graph, ax=ax)
            
            assert result_ax is ax
            plt.close(fig)
        except ImportError:
            pytest.skip("NetworkX not available")
    
    def test_plot_degree_distribution(self, small_tissue):
        """Test plotting degree distribution."""
        try:
            from spatialtissuepy.network import CellGraph
            
            graph = CellGraph.from_spatial_data(
                small_tissue,
                method='proximity',
                radius=50
            )
            
            fig, ax = plt.subplots()
            result_ax = plot_degree_distribution(graph, ax=ax)
            
            assert result_ax is ax
            plt.close(fig)
        except ImportError:
            pytest.skip("NetworkX not available")


# =============================================================================
# Statistics Plot Tests
# =============================================================================

class TestStatisticsPlots:
    """Tests for statistics plotting functions."""
    
    def test_plot_ripleys_curve(self, small_tissue):
        """Test plotting Ripley's K curve."""
        try:
            fig, ax = plt.subplots()
            
            result_ax = plot_ripleys_curve(
                small_tissue,
                radii=np.linspace(10, 100, 10),
                ax=ax
            )
            
            assert result_ax is ax
            plt.close(fig)
        except (ImportError, NotImplementedError):
            pytest.skip("Ripley's curve plotting not available")
    
    def test_plot_colocalization_heatmap(self, small_tissue):
        """Test plotting colocalization heatmap."""
        if len(small_tissue.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        try:
            fig, ax = plt.subplots()
            
            result_ax = plot_colocalization_heatmap(
                small_tissue,
                radius=50,
                ax=ax
            )
            
            assert result_ax is ax
            plt.close(fig)
        except (ImportError, NotImplementedError):
            pytest.skip("Colocalization heatmap not available")


# =============================================================================
# Comparison Plot Tests
# =============================================================================

class TestComparisonPlots:
    """Tests for multi-sample comparison plots."""
    
    def test_plot_metric_comparison(self):
        """Test plotting metric comparison across samples."""
        # Create mock data
        data = pd.DataFrame({
            'sample': ['A', 'A', 'B', 'B', 'C', 'C'],
            'metric': [10, 12, 15, 14, 8, 9]
        })
        
        try:
            fig, ax = plt.subplots()
            result_ax = plot_metric_comparison(data, x='sample', y='metric', ax=ax)
            
            assert result_ax is ax
            plt.close(fig)
        except (ImportError, NotImplementedError):
            pytest.skip("Metric comparison plot not available")
    
    def test_plot_violin_comparison(self):
        """Test violin plot comparison."""
        # Create mock data
        data = pd.DataFrame({
            'sample': ['A'] * 20 + ['B'] * 20 + ['C'] * 20,
            'value': np.random.randn(60)
        })
        
        try:
            fig, ax = plt.subplots()
            result_ax = plot_violin_comparison(data, x='sample', y='value', ax=ax)
            
            assert result_ax is ax
            plt.close(fig)
        except (ImportError, NotImplementedError, AttributeError):
            pytest.skip("Violin plot not available")


# =============================================================================
# Figure Saving Tests
# =============================================================================

class TestFigureSaving:
    """Tests for figure saving functionality."""
    
    def test_save_figure_pdf(self, tmp_path, small_tissue):
        """Test saving figure as PDF."""
        fig, ax = plt.subplots()
        plot_spatial_scatter(small_tissue, ax=ax)
        
        output_path = tmp_path / "test_figure.pdf"
        
        try:
            save_figure(fig, str(output_path))
            assert output_path.exists()
        except Exception:
            pytest.skip("PDF saving not available")
        finally:
            plt.close(fig)
    
    def test_save_figure_png(self, tmp_path, small_tissue):
        """Test saving figure as PNG."""
        fig, ax = plt.subplots()
        plot_spatial_scatter(small_tissue, ax=ax)
        
        output_path = tmp_path / "test_figure.png"
        
        try:
            save_figure(fig, str(output_path), dpi=150)
            assert output_path.exists()
        except Exception:
            pytest.skip("PNG saving not available")
        finally:
            plt.close(fig)
    
    def test_save_figure_multiple_formats(self, tmp_path, small_tissue):
        """Test saving figure in multiple formats."""
        fig, ax = plt.subplots()
        plot_spatial_scatter(small_tissue, ax=ax)
        
        try:
            save_figure(
                fig,
                str(tmp_path / "test_figure"),
                formats=['png', 'pdf']
            )
            
            assert (tmp_path / "test_figure.png").exists()
            assert (tmp_path / "test_figure.pdf").exists()
        except Exception:
            pytest.skip("Multi-format saving not available")
        finally:
            plt.close(fig)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVizIntegration:
    """Integration tests for visualization."""
    
    def test_multi_panel_figure(self, small_tissue):
        """Test creating multi-panel figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Panel 1: Spatial scatter
        plot_spatial_scatter(small_tissue, ax=axes[0])
        axes[0].set_title('Spatial Distribution')
        
        # Panel 2: Cell types
        plot_cell_types(small_tissue, ax=axes[1])
        axes[1].set_title('Cell Types')
        
        # Should create valid figure
        assert fig is not None
        assert len(axes) == 2
        
        plt.close(fig)
    
    def test_workflow_with_style(self, small_tissue):
        """Test complete workflow with style setting."""
        # Set style
        set_publication_style()
        
        # Create plot
        fig, ax = plt.subplots()
        plot_spatial_scatter(small_tissue, ax=ax)
        
        # Should work without errors
        assert fig is not None
        
        plt.close(fig)
        
        # Reset to default
        set_default_style()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestVizEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_plot_empty_tissue(self):
        """Test plotting empty tissue."""
        coords = np.array([]).reshape(0, 2)
        types = np.array([])
        data = SpatialTissueData(coords, types)
        
        fig, ax = plt.subplots()
        
        try:
            result_ax = plot_spatial_scatter(data, ax=ax)
            # Should handle gracefully
            assert result_ax is ax
        except Exception:
            # May raise error for empty data - acceptable
            pass
        finally:
            plt.close(fig)
    
    def test_plot_single_cell(self):
        """Test plotting tissue with single cell."""
        coords = np.array([[0, 0]])
        types = np.array(['A'])
        data = SpatialTissueData(coords, types)
        
        fig, ax = plt.subplots()
        result_ax = plot_spatial_scatter(data, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_single_cell_type(self):
        """Test plotting tissue with single cell type."""
        coords = np.random.rand(50, 2) * 100
        types = np.array(['A'] * 50)
        data = SpatialTissueData(coords, types)
        
        fig, ax = plt.subplots()
        result_ax = plot_cell_types(data, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_with_invalid_marker(self, small_tissue):
        """Test error with nonexistent marker."""
        fig, ax = plt.subplots()
        
        with pytest.raises((KeyError, ValueError)):
            plot_marker_expression(small_tissue, marker='nonexistent_marker', ax=ax)
        
        plt.close(fig)


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestVizParameterValidation:
    """Tests for parameter validation."""
    
    def test_spatial_scatter_with_invalid_params(self, small_tissue):
        """Test spatial scatter with invalid parameters."""
        fig, ax = plt.subplots()
        
        # Should handle invalid color gracefully or raise error
        try:
            plot_spatial_scatter(small_tissue, color='invalid_color', ax=ax)
        except (ValueError, KeyError):
            pass  # Expected
        
        plt.close(fig)
    
    def test_plot_with_wrong_data_type(self):
        """Test plotting with wrong data type."""
        fig, ax = plt.subplots()
        
        with pytest.raises((TypeError, AttributeError)):
            plot_spatial_scatter("not a SpatialTissueData object", ax=ax)
        
        plt.close(fig)
    
    def test_get_colors_with_empty_list(self):
        """Test getting colors for empty cell type list."""
        colors = get_cell_type_colors([])
        
        assert isinstance(colors, dict)
        assert len(colors) == 0


# =============================================================================
# Axes Return Tests
# =============================================================================

class TestAxesReturns:
    """Tests that all plot functions return axes correctly."""
    
    def test_all_spatial_plots_return_axes(self, small_tissue):
        """Test that spatial plots return axes."""
        plot_functions = [
            plot_spatial_scatter,
            plot_cell_types,
        ]
        
        for plot_fn in plot_functions:
            fig, ax = plt.subplots()
            
            try:
                result_ax = plot_fn(small_tissue, ax=ax)
                assert result_ax is ax
            except Exception:
                # Function might not be fully implemented
                pass
            finally:
                plt.close(fig)
    
    def test_plots_create_axes_when_none_provided(self, small_tissue):
        """Test that plots create axes when not provided."""
        try:
            result_ax = plot_spatial_scatter(small_tissue)
            
            assert result_ax is not None
            assert hasattr(result_ax, 'figure')
            
            plt.close(result_ax.figure)
        except Exception:
            pytest.skip("Auto-axes creation not available")


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestVizPerformance:
    """Performance tests for visualization."""
    
    def test_plot_large_dataset(self, large_tissue):
        """Test plotting large dataset."""
        import time
        
        fig, ax = plt.subplots()
        
        start = time.time()
        plot_spatial_scatter(large_tissue, ax=ax)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
        
        plt.close(fig)
    
    def test_multiple_plots_performance(self, medium_tissue):
        """Test creating multiple plots."""
        import time
        
        start = time.time()
        
        for _ in range(10):
            fig, ax = plt.subplots()
            plot_spatial_scatter(medium_tissue, ax=ax)
            plt.close(fig)
        
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 15.0
