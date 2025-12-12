"""
Spatial LDA metrics for integration with summary module.

This module registers LDA-derived metrics with the StatisticsPanel
for standardized computation across samples.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from spatialtissuepy.summary.registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# Store fitted models for reuse within a session
_model_cache: Dict[str, Any] = {}


def _get_or_fit_model(
    data: 'SpatialTissueData',
    n_topics: int,
    radius: float,
    cache_key: str
) -> 'SpatialLDA':
    """Get cached model or fit a new one."""
    from .spatial_lda import SpatialLDA
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    _model_cache[cache_key] = model
    
    return model


# -----------------------------------------------------------------------------
# Topic Distribution Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='lda_topic_proportions',
    category='lda',
    description='Proportion of cells assigned to each topic',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_topic_proportions(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute topic proportions based on dominant assignment."""
    from .spatial_lda import SpatialLDA
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    dominant = model.predict(data)
    
    result = {}
    for i in range(n_topics):
        result[f'topic_{i}_proportion'] = float(np.mean(dominant == i))
    
    return result


@register_metric(
    name='lda_topic_entropy',
    category='lda',
    description='Mean entropy of topic assignments (uncertainty)',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_topic_entropy(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute mean topic assignment entropy."""
    from .spatial_lda import SpatialLDA
    from .analysis import topic_assignment_uncertainty
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    entropy = topic_assignment_uncertainty(model, data)
    
    return {
        'lda_mean_entropy': float(np.mean(entropy)),
        'lda_max_entropy': float(np.max(entropy)),
    }


@register_metric(
    name='lda_dominant_topic_confidence',
    category='lda',
    description='Mean confidence of dominant topic assignments',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_dominant_confidence(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute mean confidence of dominant topic."""
    from .spatial_lda import SpatialLDA
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    weights = model.transform(data)
    confidence = np.max(weights, axis=1)
    
    return {
        'lda_mean_confidence': float(np.mean(confidence)),
        'lda_min_confidence': float(np.min(confidence)),
    }


# -----------------------------------------------------------------------------
# Topic Quality Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='lda_diversity',
    category='lda',
    description='Topic diversity score (how distinct topics are)',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_diversity(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute topic diversity."""
    from .spatial_lda import SpatialLDA
    from .metrics import topic_diversity
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    diversity = topic_diversity(model)
    
    return {'lda_diversity': diversity}


@register_metric(
    name='lda_spatial_consistency',
    category='lda',
    description='Spatial consistency of topic assignments',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_spatial_consistency(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute spatial consistency metrics."""
    from .spatial_lda import SpatialLDA
    from .metrics import spatial_topic_consistency
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    consistency = spatial_topic_consistency(model, data, radius)
    
    return {
        'lda_agreement_rate': consistency['agreement_rate'],
        'lda_topic_autocorrelation': consistency['topic_autocorrelation'],
    }


@register_metric(
    name='lda_perplexity',
    category='lda',
    description='Model perplexity (lower is better fit)',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_perplexity(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute model perplexity."""
    from .spatial_lda import SpatialLDA
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    perplexity = model.perplexity(data)
    
    return {'lda_perplexity': perplexity}


# -----------------------------------------------------------------------------
# Topic-Cell Type Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='lda_max_topic_enrichment',
    category='lda',
    description='Maximum cell type enrichment in any topic',
    parameters={'n_topics': int, 'radius': float, 'cell_type': str}
)
def _lda_max_enrichment(
    data: 'SpatialTissueData',
    cell_type: str,
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute max enrichment of a cell type across topics."""
    from .spatial_lda import SpatialLDA
    from .analysis import topic_enrichment
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    enrichment = topic_enrichment(model)
    
    if cell_type in enrichment.columns:
        max_enrich = enrichment[cell_type].max()
    else:
        max_enrich = np.nan
    
    return {f'lda_max_enrichment_{cell_type}': float(max_enrich)}


@register_metric(
    name='lda_topic_concentration',
    category='lda',
    description='Spatial concentration index for each topic',
    parameters={'n_topics': int, 'radius': float}
)
def _lda_topic_concentration(
    data: 'SpatialTissueData',
    n_topics: int = 5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute topic concentration indices."""
    from .spatial_lda import SpatialLDA
    from .metrics import topic_concentration_index
    
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=radius,
        random_state=42
    )
    model.fit(data)
    
    concentration = topic_concentration_index(model, data)
    
    result = {}
    for topic_idx, conc in concentration.items():
        result[f'lda_topic_{topic_idx}_concentration'] = float(conc)
    
    result['lda_mean_concentration'] = float(np.mean(list(concentration.values())))
    
    return result
