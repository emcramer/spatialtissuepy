=============
Visualization
=============

This guide covers creating publication-quality figures.


Configuration
-------------

Publication Styles
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import set_publication_style, set_default_style

    # Set journal style
    set_publication_style(journal='nature')  # or 'science', 'cell'

    # Reset to defaults
    set_default_style()

Custom Colors
~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import get_cell_type_colors

    # Get automatic colors
    colors = get_cell_type_colors(data.cell_types_unique)

    # Define custom colors
    custom_colors = {
        'Tumor': '#E31A1C',
        'CD8_T_cell': '#1F78B4',
        'Macrophage': '#33A02C',
        'Stromal': '#FF7F00',
    }


Spatial Plots
-------------

Basic Scatter Plot
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_spatial_scatter

    # Color by cell type
    plot_spatial_scatter(data)

    # Color by marker expression
    plot_spatial_scatter(data, marker='Ki67', cmap='magma')

    # Custom colors and styling
    plot_spatial_scatter(
        data,
        colors=custom_colors,
        size=10,
        alpha=0.8,
        title='My Tissue Sample'
    )

Cell Type Panels
~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_cell_types

    # One panel per cell type
    fig = plot_cell_types(data, ncols=3)

Marker Expression Panels
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_marker_expression

    fig = plot_marker_expression(
        data,
        markers=['CD8', 'PD1', 'Ki67'],
        ncols=3,
        cmap='magma'
    )

Density Map
~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_density_map

    # Overall density
    plot_density_map(data)

    # Density of specific type
    plot_density_map(data, cell_type='Tumor', cmap='Reds')


Statistics Plots
----------------

Ripley's Curve
~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_ripleys_curve

    plot_ripleys_curve(data, max_radius=200)

Colocalization Heatmap
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.statistics import colocalization_matrix
    from spatialtissuepy.viz import plot_colocalization_heatmap

    clq = colocalization_matrix(data, radius=50)
    plot_colocalization_heatmap(clq, annot=True)

Hotspot Map
~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_hotspot_map

    plot_hotspot_map(data, cell_type='Tumor', radius=50)


Network Plots
-------------

Cell Graph
~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.network import CellGraph
    from spatialtissuepy.viz import plot_cell_graph

    graph = CellGraph.from_spatial_data(data, method='proximity', radius=50)
    plot_cell_graph(graph, data, edge_alpha=0.3)

Degree Distribution
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_degree_distribution

    plot_degree_distribution(graph)

Type Mixing Matrix
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_type_mixing_matrix

    plot_type_mixing_matrix(graph)


LDA Plots
---------

Topic Composition
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_topic_composition

    plot_topic_composition(model)

Topic Spatial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_topic_spatial

    plot_topic_spatial(data, topic_weights, topic_id=0)


Mapper Plots
------------

.. code-block:: python

    from spatialtissuepy.viz import plot_mapper_graph, plot_mapper_spatial

    # Mapper graph
    plot_mapper_graph(result, color_by='size')

    # Spatial embedding
    plot_mapper_spatial(result, data)


Multi-Panel Figures
-------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from spatialtissuepy.viz import set_publication_style

    set_publication_style(journal='nature')

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel A: Spatial scatter
    ax_a = fig.add_subplot(gs[0, :2])
    plot_spatial_scatter(data, ax=ax_a)
    ax_a.set_title('A', loc='left', fontweight='bold')

    # Panel B: Density
    ax_b = fig.add_subplot(gs[0, 2])
    plot_density_map(data, ax=ax_b)
    ax_b.set_title('B', loc='left', fontweight='bold')

    # Panel C: Ripley's H
    ax_c = fig.add_subplot(gs[1, 0])
    plot_ripleys_curve(data, ax=ax_c)
    ax_c.set_title('C', loc='left', fontweight='bold')

    # Panel D: Colocalization
    ax_d = fig.add_subplot(gs[1, 1])
    plot_colocalization_heatmap(clq, ax=ax_d)
    ax_d.set_title('D', loc='left', fontweight='bold')

    # Panel E: Network
    ax_e = fig.add_subplot(gs[1, 2])
    plot_cell_graph(graph, data, ax=ax_e, edge_alpha=0.2)
    ax_e.set_title('E', loc='left', fontweight='bold')

    plt.tight_layout()


Saving Figures
--------------

.. code-block:: python

    from spatialtissuepy.viz import save_figure

    # Save in multiple formats
    save_figure(fig, 'figure1', formats=['pdf', 'png', 'svg'], dpi=300)


Customization Tips
------------------

Using Axes
~~~~~~~~~~

All plot functions accept an ``ax`` parameter:

.. code-block:: python

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_spatial_scatter(data, ax=ax)
    ax.set_title('My Custom Title')

Color Maps
~~~~~~~~~~

Common color maps:

- **Categorical**: tab10, tab20, Set1, Set2
- **Sequential**: viridis, plasma, magma, cividis
- **Diverging**: RdBu, coolwarm, seismic

Scalebars
~~~~~~~~~

.. code-block:: python

    plot_spatial_scatter(data, scalebar=100)  # 100µm scalebar

High Resolution
~~~~~~~~~~~~~~~

.. code-block:: python

    fig.savefig('figure.png', dpi=300, bbox_inches='tight')
    fig.savefig('figure.pdf', bbox_inches='tight')  # Vector format
