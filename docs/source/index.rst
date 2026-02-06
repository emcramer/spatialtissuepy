.. spatialtissuepy documentation master file

===============================================
spatialtissuepy: Spatial Tissue Analysis
===============================================

**spatialtissuepy** is a Python package for spatial analysis of tissue biology data
from multiplexed imaging experiments and agent-based modeling simulations.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Getting Started
        :link: getting_started/index
        :link-type: doc

        New to spatialtissuepy? Start here for installation and a quick introduction.

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc

        In-depth guides on using spatialtissuepy for spatial tissue analysis.

    .. grid-item-card:: Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step tutorials covering common analysis workflows.

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Complete reference documentation for all modules and functions.


Key Features
------------

.. grid:: 3
    :gutter: 2

    .. grid-item::

        **Spatial Statistics**

        Ripley's K/L/H functions, colocalization analysis,
        hotspot detection, and Moran's I.

    .. grid-item::

        **Network Analysis**

        Build cell graphs from spatial data, compute centrality
        measures, and analyze cell-cell interactions.

    .. grid-item::

        **Topic Modeling**

        Discover cellular microenvironments using Spatial LDA
        for neighborhood-based topic modeling.

    .. grid-item::

        **Topological Analysis**

        Use the Mapper algorithm to discover tissue communities
        and spatial organization patterns.

    .. grid-item::

        **Feature Extraction**

        Extract interpretable features for machine learning
        from spatial tissue data.

    .. grid-item::

        **ABM Integration**

        Load and analyze agent-based model outputs from
        PhysiCell and other simulators.


Quick Example
-------------

.. code-block:: python

    import numpy as np
    from spatialtissuepy import SpatialTissueData
    from spatialtissuepy.statistics import ripleys_h, colocalization_quotient
    from spatialtissuepy.viz import plot_spatial_scatter

    # Create spatial tissue data
    coordinates = np.random.uniform(0, 1000, (500, 2))
    cell_types = np.random.choice(['Tumor', 'T_cell', 'Stromal'], 500)

    data = SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types
    )

    # Compute spatial statistics
    radii = np.linspace(10, 200, 20)
    H = ripleys_h(data, radii=radii)

    clq = colocalization_quotient(
        data,
        type_a='Tumor',
        type_b='T_cell',
        radius=50
    )

    # Visualize
    plot_spatial_scatter(data)


Installation
------------

Install spatialtissuepy using pip:

.. code-block:: bash

    pip install spatialtissuepy

Or with conda:

.. code-block:: bash

    conda install -c conda-forge spatialtissuepy

See the :doc:`getting_started/installation` guide for detailed instructions.


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   tutorials/index
   api/index
   changelog
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
