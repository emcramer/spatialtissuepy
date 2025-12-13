"""
Visualization module for spatialtissuepy.

Provides publication-quality plotting tools for spatial tissue analysis,
designed for seamless integration with matplotlib for multi-panel figures.

Submodules
----------
config : Plot configuration, themes, and style settings
spatial : Spatial scatter plots and cell visualizations
network : Graph and network visualizations
statistics : Statistical plots (Ripley's curves, heatmaps)
comparison : Multi-sample comparison plots (violin, heatmaps)
lda : Spatial LDA topic visualizations
mapper : Mapper/TDA visualizations
qc : Quality control and diagnostic plots

Design Philosophy
-----------------
All plotting functions follow these conventions:
1. Accept optional `ax` parameter for embedding in multi-panel figures
2. Return the matplotlib Axes object for further customization
3. Use consistent color palettes and styling
4. Support both quick exploratory plots and publication-quality output

Example: Creating a Multi-Panel Figure
--------------------------------------
>>> import matplotlib.pyplot as plt
>>> from spatialtissuepy.viz import (
...     plot_spatial_scatter, plot_ripleys_curve, plot_cell_type_heatmap
... )
>>>
>>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
>>> plot_spatial_scatter(data, ax=axes[0])
>>> plot_ripleys_curve(data, ax=axes[1])
>>> plot_cell_type_heatmap(results_df, ax=axes[2])
>>> fig.tight_layout()
>>> fig.savefig('figure1.pdf', dpi=300)

Publication Export
------------------
>>> from spatialtissuepy.viz import set_publication_style, save_figure
>>> set_publication_style()  # Configure for publication
>>> fig = create_my_figure()
>>> save_figure(fig, 'figure1', formats=['pdf', 'png', 'svg'])
"""

# Configuration and themes
from .config import (
    set_publication_style,
    set_default_style,
    get_cell_type_colors,
    get_categorical_palette,
    get_sequential_cmap,
    get_diverging_cmap,
    save_figure,
    PlotConfig,
)

# Spatial plots
from .spatial import (
    plot_spatial_scatter,
    plot_cell_types,
    plot_marker_expression,
    plot_density_map,
    plot_voronoi,
    plot_spatial_domains,
    plot_cell_neighborhoods,
)

# Network plots
from .network import (
    plot_cell_graph,
    plot_graph_on_tissue,
    plot_degree_distribution,
    plot_centrality_by_type,
    plot_type_mixing_matrix,
)

# Statistics plots
from .statistics import (
    plot_ripleys_curve,
    plot_pcf_curve,
    plot_colocalization_heatmap,
    plot_neighborhood_enrichment,
    plot_hotspot_map,
    plot_morans_scatter,
)

# Comparison plots
from .comparison import (
    plot_metric_comparison,
    plot_metric_heatmap,
    plot_violin_comparison,
    plot_pca_samples,
    plot_umap_samples,
    plot_sample_correlation,
    plot_trajectory,
)

# LDA plots
from .lda import (
    plot_topic_composition,
    plot_topic_spatial,
    plot_topic_enrichment_heatmap,
    plot_topic_transition_matrix,
    plot_lda_diagnostics,
    plot_topic_proportions_bar,
)

# Mapper/TDA plots
from .mapper import (
    plot_mapper_graph,
    plot_mapper_spatial,
    plot_filter_distribution,
    plot_node_composition,
    plot_mapper_diagnostics,
    create_mapper_report,
)

# Quality control plots
from .qc import (
    plot_cell_count_summary,
    plot_spatial_coverage,
    plot_model_selection,
    plot_stability_analysis,
    plot_convergence,
)

__all__ = [
    # Config
    'set_publication_style',
    'set_default_style',
    'get_cell_type_colors',
    'get_categorical_palette',
    'get_sequential_cmap',
    'get_diverging_cmap',
    'save_figure',
    'PlotConfig',
    # Spatial
    'plot_spatial_scatter',
    'plot_cell_types',
    'plot_marker_expression',
    'plot_density_map',
    'plot_voronoi',
    'plot_spatial_domains',
    'plot_cell_neighborhoods',
    # Network
    'plot_cell_graph',
    'plot_graph_on_tissue',
    'plot_degree_distribution',
    'plot_centrality_by_type',
    'plot_type_mixing_matrix',
    # Statistics
    'plot_ripleys_curve',
    'plot_pcf_curve',
    'plot_colocalization_heatmap',
    'plot_neighborhood_enrichment',
    'plot_hotspot_map',
    'plot_morans_scatter',
    # Comparison
    'plot_metric_comparison',
    'plot_metric_heatmap',
    'plot_violin_comparison',
    'plot_pca_samples',
    'plot_umap_samples',
    'plot_sample_correlation',
    'plot_trajectory',
    # LDA
    'plot_topic_composition',
    'plot_topic_spatial',
    'plot_topic_enrichment_heatmap',
    'plot_topic_transition_matrix',
    'plot_lda_diagnostics',
    'plot_topic_proportions_bar',
    # Mapper
    'plot_mapper_graph',
    'plot_mapper_spatial',
    'plot_filter_distribution',
    'plot_node_composition',
    'plot_mapper_diagnostics',
    'create_mapper_report',
    # QC
    'plot_cell_count_summary',
    'plot_spatial_coverage',
    'plot_model_selection',
    'plot_stability_analysis',
    'plot_convergence',
]
