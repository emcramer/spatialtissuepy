==============
Topic Modeling
==============

This guide covers Spatial LDA for discovering cellular microenvironments.


What is Spatial LDA?
--------------------

Spatial LDA (Latent Dirichlet Allocation) discovers recurrent patterns of
cell type compositions in neighborhoods. Each pattern is called a "topic"
and represents a distinct microenvironment.

**Analogy to text:**

- Documents → Cell neighborhoods
- Words → Cell types in neighborhood
- Topics → Microenvironment patterns


Basic Usage
-----------

.. code-block:: python

    from spatialtissuepy.lda import SpatialLDA, fit_spatial_lda

    # Quick fit
    model = fit_spatial_lda(
        data,
        n_topics=5,
        neighborhood_radius=50,
        random_state=42
    )

    # Get topic weights for each cell
    topic_weights = model.transform(data)  # (n_cells, n_topics)

    # Get dominant topic assignment
    dominant_topic = model.predict(data)  # (n_cells,)


Choosing Parameters
-------------------

Number of Topics
~~~~~~~~~~~~~~~~

The optimal number of topics depends on tissue complexity:

- **3-5 topics**: Simple tissues with few microenvironments
- **5-10 topics**: Typical for tumor microenvironment studies
- **10+ topics**: Complex tissues or multi-sample studies

Use model selection:

.. code-block:: python

    from spatialtissuepy.lda import SpatialLDA

    # Try different numbers of topics
    perplexities = []
    for n_topics in range(3, 12):
        model = SpatialLDA(n_topics=n_topics, random_state=42)
        model.fit(data)
        perplexities.append(model.perplexity(data))

    # Plot elbow curve
    import matplotlib.pyplot as plt
    plt.plot(range(3, 12), perplexities, 'o-')
    plt.xlabel('Number of topics')
    plt.ylabel('Perplexity (lower is better)')
    plt.show()

Neighborhood Radius
~~~~~~~~~~~~~~~~~~~

- **Small radius (20-30µm)**: Local interactions, cell-cell contact
- **Medium radius (50-100µm)**: Typical for microenvironment analysis
- **Large radius (100-200µm)**: Regional patterns


Interpreting Topics
-------------------

Topic Summary
~~~~~~~~~~~~~

.. code-block:: python

    # Get topic-cell type matrix
    summary = model.topic_summary()
    print(summary)

    # Top cell types per topic
    top_types = model.top_cell_types_per_topic(n_top=3)

    for topic_id, types_weights in top_types.items():
        print(f"\nTopic {topic_id}:")
        for cell_type, weight in types_weights:
            print(f"  {cell_type}: {weight:.3f}")

Naming Topics
~~~~~~~~~~~~~

Based on composition, give topics biological names:

.. code-block:: python

    topic_names = {
        0: 'Tumor core',
        1: 'Immune infiltrate',
        2: 'Stromal border',
        3: 'Tumor-immune interface',
        4: 'Background'
    }


Visualization
-------------

Topic Spatial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_topic_spatial, plot_topic_composition

    # Show topic weights in space
    plot_topic_spatial(data, topic_weights, topic_id=0)

    # Show topic composition as stacked bar
    plot_topic_composition(model)

All Topics
~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, topic_id in zip(axes.flat, range(model.n_topics)):
        scatter = ax.scatter(
            data.coordinates[:, 0],
            data.coordinates[:, 1],
            c=topic_weights[:, topic_id],
            cmap='Reds', s=5, alpha=0.7
        )
        ax.set_title(f'Topic {topic_id}')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


Adding Topics to Data
---------------------

.. code-block:: python

    # Add topic weights as new columns
    data_with_topics = model.add_topics_to_data(data, prefix='topic')

    # Now data has topic_0, topic_1, ..., topic_dominant columns
    print(data_with_topics.marker_names)


Multi-Sample Analysis
---------------------

Fit on Multiple Samples
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit on cohort
    samples = [data1, data2, data3, data4]
    model = SpatialLDA(n_topics=5, neighborhood_radius=50)
    model.fit(samples)

    # Transform each sample
    for sample in samples:
        topic_weights = model.transform(sample)
        # Analyze...

Topic Proportions per Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sample_topic_props = []

    for sample, sample_id in zip(samples, sample_ids):
        weights = model.transform(sample)
        # Average topic weights across cells
        mean_weights = weights.mean(axis=0)
        sample_topic_props.append({
            'sample_id': sample_id,
            **{f'topic_{i}': w for i, w in enumerate(mean_weights)}
        })

    import pandas as pd
    topic_df = pd.DataFrame(sample_topic_props)
    print(topic_df)


Quality Metrics
---------------

.. code-block:: python

    from spatialtissuepy.lda import topic_coherence, topic_diversity

    # Topic coherence (higher is better)
    coherence = topic_coherence(model, data)
    print(f"Coherence: {coherence:.4f}")

    # Topic diversity (higher = more distinct topics)
    diversity = topic_diversity(model)
    print(f"Diversity: {diversity:.4f}")


Biological Interpretation
-------------------------

Common Topic Patterns
~~~~~~~~~~~~~~~~~~~~~

1. **Tumor core**: High tumor cell proportion, low immune
2. **Immune infiltrate**: High T cell, macrophage proportions
3. **Stromal compartment**: High fibroblast, endothelial
4. **Tumor-immune interface**: Mixed tumor and immune
5. **Tertiary lymphoid structures**: B cells, T cells, DCs

Downstream Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Correlate topics with outcomes
    # Example: survival analysis

    # Get per-patient topic proportions
    patient_topics = []
    for patient_id, sample in patient_samples.items():
        weights = model.transform(sample).mean(axis=0)
        patient_topics.append({
            'patient_id': patient_id,
            'topic_immune': weights[1],  # Immune infiltrate topic
            'survival_months': patient_survival[patient_id]
        })

    # Now correlate topic_immune with survival
