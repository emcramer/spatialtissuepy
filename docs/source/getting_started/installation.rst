============
Installation
============

This guide covers how to install spatialtissuepy and its dependencies.


Requirements
------------

spatialtissuepy requires:

- Python 3.9 or later
- NumPy >= 1.20
- SciPy >= 1.7
- pandas >= 1.3
- scikit-learn >= 1.0
- matplotlib >= 3.4


Quick Install
-------------

The simplest way to install spatialtissuepy is using pip:

.. code-block:: bash

    pip install spatialtissuepy


Install with Optional Dependencies
----------------------------------

spatialtissuepy has optional dependencies for additional functionality:

**Network analysis** (requires NetworkX):

.. code-block:: bash

    pip install spatialtissuepy[network]

**All optional dependencies**:

.. code-block:: bash

    pip install spatialtissuepy[all]

This includes:

- ``networkx``: For graph-based analysis
- ``tqdm``: For progress bars
- ``seaborn``: For enhanced visualizations


Development Installation
------------------------

To install for development:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/yourusername/spatialtissuepy.git
    cd spatialtissuepy

    # Create a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install in editable mode with dev dependencies
    pip install -e ".[dev]"


Conda Installation
------------------

If you use conda:

.. code-block:: bash

    conda install -c conda-forge spatialtissuepy

Or create a new environment:

.. code-block:: bash

    conda create -n spatial python=3.11
    conda activate spatial
    pip install spatialtissuepy


Verifying Installation
----------------------

To verify the installation:

.. code-block:: python

    import spatialtissuepy
    print(spatialtissuepy.__version__)

    # Quick test
    from spatialtissuepy import SpatialTissueData
    import numpy as np

    data = SpatialTissueData(
        coordinates=np.random.rand(100, 2) * 1000,
        cell_types=np.array(['A'] * 50 + ['B'] * 50)
    )
    print(data)


Troubleshooting
---------------

**ImportError: No module named 'spatialtissuepy'**

Make sure you've activated the correct Python environment and that
the package is installed.

**NetworkX not available**

Some network analysis functions require NetworkX. Install it with:

.. code-block:: bash

    pip install networkx

**Memory issues with large datasets**

For very large datasets (>100,000 cells), consider:

- Using chunked processing
- Reducing neighborhood radii
- Downsampling for exploratory analysis
