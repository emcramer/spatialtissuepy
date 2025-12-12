"""
Spatial LDA module for discovering recurrent cellular neighborhoods.

This module implements Latent Dirichlet Allocation (LDA) adapted for spatial
tissue analysis. It identifies "topics" (recurrent cellular neighborhood patterns)
that characterize tissue microenvironments.

Key Concepts
------------
- **Topic**: A distribution over cell types that characterizes a microenvironment
- **Neighborhood**: The local cellular context around each cell
- **Document**: In spatial LDA, each cell's neighborhood is treated as a document
- **Words**: Cell types in the neighborhood are the "words"

The method discovers recurrent spatial patterns by:
1. Computing neighborhood composition for each cell
2. Fitting LDA to find topics (characteristic cell type mixtures)
3. Assigning topic weights to each cell

This enables:
- Discovery of tissue microenvironment signatures
- Comparison of spatial organization across samples
- Feature extraction for downstream ML tasks

References
----------
.. [1] Chen, Z. et al. (2020). Modeling Multiplexed Images with Spatial-LDA
       Reveals Novel Tissue Microenvironments. J Comput Biol.
.. [2] Blei, D. M. et al. (2003). Latent Dirichlet Allocation. JMLR.
.. [3] Keren, L. et al. (2018). A Structured Tumor-Immune Microenvironment in
       Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging. Cell.
"""

from .spatial_lda import (
    SpatialLDA,
    fit_spatial_lda,
    compute_neighborhood_features,
)

from .sampling import (
    poisson_disk_sample,
    grid_sample,
    random_sample,
    stratified_sample,
)

from .analysis import (
    topic_cell_type_matrix,
    topic_enrichment,
    dominant_topic_per_cell,
    topic_spatial_distribution,
    compare_topics_across_samples,
)

from .metrics import (
    topic_coherence,
    topic_diversity,
    spatial_topic_consistency,
)

__all__ = [
    # Main class
    'SpatialLDA',
    'fit_spatial_lda',
    'compute_neighborhood_features',
    # Sampling
    'poisson_disk_sample',
    'grid_sample',
    'random_sample',
    'stratified_sample',
    # Analysis
    'topic_cell_type_matrix',
    'topic_enrichment',
    'dominant_topic_per_cell',
    'topic_spatial_distribution',
    'compare_topics_across_samples',
    # Metrics
    'topic_coherence',
    'topic_diversity',
    'spatial_topic_consistency',
]
