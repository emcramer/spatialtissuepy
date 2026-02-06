==================
Spatial Statistics
==================

This guide covers the spatial statistics functions in spatialtissuepy.


Ripley's Functions
------------------

Ripley's K, L, and H functions measure spatial clustering or dispersion.

Ripley's K
~~~~~~~~~~

K(r) is the expected number of cells within distance r of a typical cell,
normalized by density:

.. code-block:: python

    from spatialtissuepy.statistics import ripleys_k
    import numpy as np

    radii = np.linspace(10, 200, 20)
    K = ripleys_k(data, radii=radii)

    # For complete spatial randomness (CSR):
    # K(r) = π * r²

Ripley's L
~~~~~~~~~~

L(r) is a variance-stabilized transformation of K:

.. code-block:: python

    from spatialtissuepy.statistics import ripleys_l

    L = ripleys_l(data, radii=radii)

    # For CSR: L(r) = r

Ripley's H
~~~~~~~~~~

H(r) = L(r) - r, centered so H = 0 for CSR:

.. code-block:: python

    from spatialtissuepy.statistics import ripleys_h

    H = ripleys_h(data, radii=radii)

    # Interpretation:
    # H > 0: Clustering (cells closer than expected)
    # H < 0: Dispersion (cells farther than expected)
    # H ≈ 0: Random (CSR)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(radii, H, 'b-', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', label='CSR')
    plt.xlabel('Radius (µm)')
    plt.ylabel('H(r)')
    plt.title("Ripley's H Function")
    plt.legend()
    plt.show()


Colocalization Analysis
-----------------------

Colocalization measures whether cell types are found together more or less
than expected by chance.

Colocalization Quotient (CLQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.statistics import colocalization_quotient

    clq = colocalization_quotient(
        data,
        type_a='CD8_T_cell',
        type_b='Tumor',
        radius=50
    )

    # Interpretation:
    # CLQ > 1: Colocalization (more neighbors than expected)
    # CLQ < 1: Avoidance (fewer neighbors than expected)
    # CLQ = 1: Random mixing

Colocalization Matrix
~~~~~~~~~~~~~~~~~~~~~

Compute CLQ for all pairs:

.. code-block:: python

    from spatialtissuepy.statistics import colocalization_matrix

    clq_matrix = colocalization_matrix(data, radius=50)
    print(clq_matrix)

    # Visualize
    import seaborn as sns
    sns.heatmap(clq_matrix, annot=True, cmap='RdBu_r', center=1)


Hotspot Detection
-----------------

Identify statistically significant clusters of high or low values.

Getis-Ord Gi*
~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.statistics import getis_ord_gi_star, cell_type_hotspots

    # For marker expression
    marker_values = data.markers['Ki67'].values
    gi_star = getis_ord_gi_star(data, marker_values, radius=50)

    # Significant hotspots (z > 1.96 for p < 0.05)
    hotspots = np.where(gi_star > 1.96)[0]
    coldspots = np.where(gi_star < -1.96)[0]

Cell Type Hotspots
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result = cell_type_hotspots(
        data,
        cell_type='Tumor',
        radius=50,
        alpha=0.05
    )

    print(f"Hotspots: {len(result['hotspot_idx'])} cells")
    print(f"Coldspots: {len(result['coldspot_idx'])} cells")

Local Moran's I
~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.statistics import local_morans_i

    result = local_morans_i(data, marker_values, radius=50)

    # LISA quadrants:
    # 1 (HH): High value, high neighbors (cluster)
    # 2 (LH): Low value, high neighbors (outlier)
    # 3 (LL): Low value, low neighbors (cluster)
    # 4 (HL): High value, low neighbors (outlier)

    hh_cells = np.where(result['quadrant'] == 1)[0]


Global Statistics
-----------------

Moran's I
~~~~~~~~~

Global measure of spatial autocorrelation:

.. code-block:: python

    from spatialtissuepy.statistics import morans_i

    I = morans_i(data, marker_values, radius=50)

    # Interpretation:
    # I > 0: Positive autocorrelation (similar values cluster)
    # I < 0: Negative autocorrelation (dissimilar values cluster)
    # I ≈ 0: Random

Clark-Evans Index
~~~~~~~~~~~~~~~~~

Ratio of observed to expected nearest-neighbor distance:

.. code-block:: python

    from spatialtissuepy.statistics import clark_evans_index

    R = clark_evans_index(data)

    # R < 1: Clustering
    # R > 1: Dispersion
    # R = 1: Random


Edge Correction
---------------

For accurate statistics near boundaries:

.. code-block:: python

    H = ripleys_h(
        data,
        radii=radii,
        edge_correction='ripley'  # or 'none', 'border'
    )


Per-Cell-Type Analysis
----------------------

Analyze each cell type separately:

.. code-block:: python

    for cell_type in data.cell_types_unique:
        subset = data.get_cells_by_type(cell_type)
        if subset.n_cells > 10:
            H = ripleys_h(subset, radii=radii)
            print(f"{cell_type}: max H = {H.max():.2f}")


Statistical Significance
------------------------

Use permutation tests:

.. code-block:: python

    from scipy import stats

    # Compute observed statistic
    observed_clq = colocalization_quotient(data, 'A', 'B', radius=50)

    # Permutation test
    n_permutations = 999
    null_distribution = []

    for _ in range(n_permutations):
        # Shuffle cell type labels
        shuffled_types = np.random.permutation(data.cell_types)
        shuffled_data = SpatialTissueData(
            coordinates=data.coordinates,
            cell_types=shuffled_types
        )
        null_clq = colocalization_quotient(shuffled_data, 'A', 'B', radius=50)
        null_distribution.append(null_clq)

    # P-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_clq))
