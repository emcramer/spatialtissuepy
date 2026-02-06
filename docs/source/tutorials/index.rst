=========
Tutorials
=========

Step-by-step tutorials covering common analysis workflows with spatialtissuepy.

Each tutorial is a Jupyter notebook that you can download and run locally.


Getting Started
---------------

.. toctree::
   :maxdepth: 1

   ../tutorials/01_quickstart
   ../tutorials/02_data_loading


Core Analysis
-------------

.. toctree::
   :maxdepth: 1

   ../tutorials/03_spatial_analysis
   ../tutorials/04_statistics
   ../tutorials/05_neighborhoods


Advanced Analysis
-----------------

.. toctree::
   :maxdepth: 1

   ../tutorials/06_networks
   ../tutorials/07_spatial_lda
   ../tutorials/08_topology


Visualization & Output
----------------------

.. toctree::
   :maxdepth: 1

   ../tutorials/09_visualization
   ../tutorials/10_multi_sample


Specialized Workflows
---------------------

.. toctree::
   :maxdepth: 1

   ../tutorials/11_physicell_integration
   ../tutorials/12_advanced_workflows


Tutorial Overview
-----------------

.. list-table::
   :widths: 5 30 15 50
   :header-rows: 1

   * - #
     - Title
     - Duration
     - Description
   * - 1
     - Quickstart
     - 5-10 min
     - Basic usage and first analysis
   * - 2
     - Data Loading
     - 10-15 min
     - Loading CSV/JSON, multi-sample data
   * - 3
     - Spatial Analysis
     - 15-20 min
     - Distances, neighborhoods, queries
   * - 4
     - Statistics
     - 20-25 min
     - Ripley's functions, colocalization, hotspots
   * - 5
     - Neighborhoods
     - 20-25 min
     - Composition, entropy, enrichment
   * - 6
     - Networks
     - 20-25 min
     - Cell graphs, centrality, assortativity
   * - 7
     - Spatial LDA
     - 25-30 min
     - Topic modeling for microenvironments
   * - 8
     - Topology
     - 25-30 min
     - Mapper algorithm, TDA
   * - 9
     - Visualization
     - 20-25 min
     - Publication-quality figures
   * - 10
     - Multi-Sample
     - 25-30 min
     - Cohort analysis, feature extraction
   * - 11
     - PhysiCell
     - 25-30 min
     - Tumor-immune ABM analysis with real simulation data
   * - 12
     - Advanced
     - 30-40 min
     - Complete analysis pipelines


Running Tutorials Locally
-------------------------

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/yourusername/spatialtissuepy.git
       cd spatialtissuepy

2. Install dependencies:

   .. code-block:: bash

       pip install -e ".[tutorials]"

3. Launch Jupyter:

   .. code-block:: bash

       jupyter notebook docs/tutorials/

4. Open any tutorial notebook and run the cells.


Prerequisites
-------------

Before starting the tutorials, ensure you have:

- Python 3.9 or later
- spatialtissuepy installed
- Jupyter notebook or JupyterLab
- Basic Python and NumPy knowledge
