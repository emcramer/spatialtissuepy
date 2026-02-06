==================
Feature Extraction
==================

This guide covers extracting features from spatial tissue data for ML.


Overview
--------

spatialtissuepy provides a systematic way to extract interpretable features
from spatial tissue samples. These features can be used for:

- Sample classification (e.g., responder vs non-responder)
- Outcome prediction (e.g., survival)
- Clustering and stratification
- Comparative analysis


Statistics Panels
-----------------

A :class:`~spatialtissuepy.summary.StatisticsPanel` defines which metrics
to compute.

Creating a Panel
~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.summary import StatisticsPanel

    panel = StatisticsPanel(name='my_analysis')

    # Add metrics
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('mean_nearest_neighbor_distance')
    panel.add('ripleys_h_max', max_radius=100)
    panel.add('colocalization_quotient', type_a='Tumor', type_b='T_cell', radius=50)

    print(panel)

Predefined Panels
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.summary import load_panel, list_panels

    # See available panels
    print(list_panels())  # ['basic', 'spatial', 'neighborhood', 'comprehensive']

    # Load a panel
    panel = load_panel('comprehensive')


Available Metrics
-----------------

.. code-block:: python

    from spatialtissuepy.summary import list_metrics, list_categories

    # See all categories
    print(list_categories())

    # See all metrics
    for metric in list_metrics():
        print(f"  {metric}")

Categories include:

- **Population**: Cell counts, proportions, diversity
- **Spatial**: Nearest neighbor distances, Ripley's functions
- **Neighborhood**: Composition, entropy, homogeneity
- **Network**: Centrality, assortativity (requires graph)
- **Morphology**: Spatial extent, convex hull


Single-Sample Summary
---------------------

.. code-block:: python

    from spatialtissuepy.summary import SpatialSummary

    summary = SpatialSummary(data, panel)

    # As dictionary
    features = summary.to_dict()
    print(features)

    # As pandas Series
    series = summary.to_series()
    print(series)

    # As numpy array (for sklearn)
    array = summary.to_array()
    print(f"Feature vector shape: {array.shape}")


Multi-Sample Summary
--------------------

For cohorts of samples:

.. code-block:: python

    from spatialtissuepy.summary import MultiSampleSummary

    # List of samples
    samples = [data1, data2, data3, data4]
    sample_ids = ['S1', 'S2', 'S3', 'S4']

    # Compute features for all
    multi_summary = MultiSampleSummary(
        samples=samples,
        panel=panel,
        sample_ids=sample_ids
    )

    # Get DataFrame
    df = multi_summary.to_dataframe()

    print(f"Feature matrix: {df.shape}")
    print(df.head())


ML Workflow
-----------

Classification Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X = df.drop(columns=['sample_id', 'label']).values
    y = df['label'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5)

    print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std()*2:.2f})")

Feature Importance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit on all data
    clf.fit(X_scaled, y)

    # Get feature importance
    importances = pd.Series(
        clf.feature_importances_,
        index=df.drop(columns=['sample_id', 'label']).columns
    ).sort_values(ascending=False)

    print("Top features:")
    print(importances.head(10))


Custom Metrics
--------------

Register custom metrics:

.. code-block:: python

    from spatialtissuepy.summary import register_metric

    def my_custom_metric(data, param1=10):
        """Compute my custom metric."""
        # Your computation here
        value = data.n_cells / param1
        return {'my_metric': value}

    register_metric(
        name='my_custom_metric',
        func=my_custom_metric,
        category='custom',
        returns=['my_metric'],
        description='My custom spatial metric'
    )

    # Now use in panel
    panel.add('my_custom_metric', param1=20)


Saving and Loading Panels
-------------------------

.. code-block:: python

    # Save panel configuration
    panel.to_json('my_panel.json')

    # Load panel
    from spatialtissuepy.summary import StatisticsPanel
    loaded_panel = StatisticsPanel.from_json('my_panel.json')


Export for Downstream Analysis
------------------------------

.. code-block:: python

    # Export feature matrix
    df.to_csv('features.csv', index=False)

    # Export with metadata
    df_with_meta = df.copy()
    df_with_meta['treatment'] = treatment_labels
    df_with_meta['outcome'] = outcomes
    df_with_meta.to_csv('features_with_metadata.csv', index=False)


Best Practices
--------------

1. **Start with predefined panels**: Use 'basic' or 'comprehensive'
2. **Add domain-specific metrics**: e.g., tumor-immune colocalization
3. **Handle missing values**: Some metrics may be NaN for edge cases
4. **Scale features**: Use StandardScaler before ML
5. **Check correlations**: Remove highly correlated features
6. **Use cross-validation**: Avoid overfitting on small cohorts
