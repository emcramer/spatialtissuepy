=============
Data Handling
=============

This guide covers how to load, manipulate, and save spatial tissue data.


Loading Data
------------

From CSV Files
~~~~~~~~~~~~~~

The most common way to load data:

.. code-block:: python

    from spatialtissuepy.io import read_csv

    data = read_csv(
        'tissue_data.csv',
        x_col='X',              # Column name for X coordinates
        y_col='Y',              # Column name for Y coordinates
        cell_type_col='CellType',  # Column name for cell types
        z_col='Z',              # Optional: for 3D data
        sample_col='SampleID',  # Optional: for multi-sample data
        marker_cols=['CD8', 'Ki67', 'PD1'],  # Optional: marker columns
    )

From JSON Files
~~~~~~~~~~~~~~~

JSON format preserves all metadata:

.. code-block:: python

    from spatialtissuepy.io import read_json

    data = read_json('tissue_data.json')

From NumPy Arrays
~~~~~~~~~~~~~~~~~

Create data programmatically:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from spatialtissuepy import SpatialTissueData

    # Coordinates and cell types
    coordinates = np.random.uniform(0, 1000, (500, 2))
    cell_types = np.random.choice(['Tumor', 'T_cell', 'Stromal'], 500)

    # Optional: marker expression
    markers = pd.DataFrame({
        'CD8': np.random.lognormal(2, 1, 500),
        'Ki67': np.random.uniform(0, 100, 500),
    })

    data = SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types,
        markers=markers,
        metadata={'experiment': 'EXP001'}
    )


Accessing Data
--------------

Basic Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Number of cells
    print(data.n_cells)

    # Unique cell types
    print(data.cell_types_unique)

    # Spatial bounds
    print(data.bounds)
    # {'x': (min, max), 'y': (min, max)}

    # Coordinates array
    coords = data.coordinates  # (n_cells, 2) array

    # Cell types array
    types = data.cell_types  # (n_cells,) array

    # Markers DataFrame
    if data.markers is not None:
        print(data.marker_names)
        cd8_values = data.markers['CD8'].values


Subsetting Data
~~~~~~~~~~~~~~~

.. code-block:: python

    # Get cells of a specific type
    tumor_data = data.get_cells_by_type('Tumor')

    # Get cells in a spatial region
    region_data = data.subset_region(
        x_min=0, x_max=500,
        y_min=0, y_max=500
    )

    # Get cells by index
    subset_data = data.subset_cells([0, 1, 2, 10, 20])

    # For multi-sample data
    sample1_data = data.subset_sample('Sample1')


Multi-Sample Data
-----------------

Loading Multiple Samples
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.io import read_csv

    # Single file with sample column
    data = read_csv(
        'cohort_data.csv',
        x_col='X', y_col='Y',
        cell_type_col='CellType',
        sample_col='SampleID'
    )

    # Check available samples
    print(data.sample_ids)

    # Access specific sample
    sample1 = data.subset_sample('Sample1')

From Multiple Files
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from spatialtissuepy.io import read_csv

    data_dir = Path('data/')
    samples = []
    sample_ids = []

    for csv_file in data_dir.glob('*.csv'):
        sample = read_csv(csv_file, x_col='X', y_col='Y', cell_type_col='CellType')
        samples.append(sample)
        sample_ids.append(csv_file.stem)


Saving Data
-----------

To CSV
~~~~~~

.. code-block:: python

    from spatialtissuepy.io import write_csv

    write_csv(data, 'output.csv')

To JSON
~~~~~~~

.. code-block:: python

    from spatialtissuepy.io import write_json

    # Preserves all metadata
    write_json(data, 'output.json')


Data Validation
---------------

spatialtissuepy validates data on creation:

.. code-block:: python

    # This will raise ValueError if data is invalid
    data = SpatialTissueData(
        coordinates=coords,
        cell_types=types
    )

    # Checks performed:
    # - coordinates must be 2D or 3D
    # - cell_types length must match n_cells
    # - markers rows must match n_cells
    # - no NaN values in coordinates


Working with Markers
--------------------

Accessing Marker Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Check if markers exist
    if data.markers is not None:
        # Get marker names
        print(data.marker_names)

        # Access specific marker
        cd8_values = data.markers['CD8'].values

        # Compute statistics
        print(f"CD8 mean: {data.markers['CD8'].mean():.2f}")

Adding Marker Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd

    # Create new markers DataFrame
    new_markers = pd.DataFrame({
        'NewMarker': np.random.uniform(0, 1, data.n_cells)
    })

    # Create new data object with markers
    new_data = SpatialTissueData(
        coordinates=data.coordinates,
        cell_types=data.cell_types,
        markers=new_markers
    )


Best Practices
--------------

1. **Use consistent coordinate units**: Usually micrometers (µm)
2. **Standardize cell type names**: Avoid spaces, use underscores
3. **Include metadata**: Store experiment info in metadata dict
4. **Validate early**: Check data quality before analysis
5. **Use JSON for reproducibility**: Preserves all information
