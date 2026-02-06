===============
Getting Started
===============

Welcome to spatialtissuepy! This section will help you get up and running
with spatial tissue analysis.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   concepts


What is spatialtissuepy?
------------------------

**spatialtissuepy** is a Python package for analyzing the spatial organization
of cells in tissue samples. It is designed for:

- **Multiplexed imaging data**: Analyze outputs from technologies like
  CODEX, MIBI-TOF, IMC, Vectra, and others.
- **Agent-based model outputs**: Analyze simulations from PhysiCell and
  similar ABM frameworks.
- **Feature extraction for ML**: Convert spatial tissue data into
  interpretable feature vectors for machine learning.


Who is this for?
----------------

spatialtissuepy is designed for:

- **Cancer biologists** studying the tumor microenvironment
- **Immunologists** analyzing immune cell organization in tissues
- **Computational biologists** building ML models on spatial data
- **Pathologists** quantifying spatial patterns in tissue sections
- **Modelers** analyzing agent-based simulation outputs


Core Concepts
-------------

The package is built around a few key concepts:

1. **SpatialTissueData**: The central data container holding cell coordinates,
   cell types, and optional marker expression data.

2. **Spatial Statistics**: Mathematical measures of spatial organization
   (clustering, dispersion, colocalization).

3. **Cell Networks**: Graph representations of cell-cell relationships.

4. **Neighborhoods**: Local microenvironments around each cell.

5. **Feature Panels**: Collections of metrics to extract from tissue samples.


Next Steps
----------

1. :doc:`installation` - Install the package
2. :doc:`quickstart` - Run your first analysis
3. :doc:`concepts` - Understand the key concepts
4. :doc:`../tutorials/index` - Follow detailed tutorials
