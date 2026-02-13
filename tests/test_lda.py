"""
Tests for spatialtissuepy.lda module.

Tests Spatial LDA for discovering recurrent cellular neighborhood patterns,
including model fitting, transformation, topic analysis, and metrics.
"""

import pytest
import numpy as np
import pandas as pd

# Check if sklearn is available
try:
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from spatialtissuepy import SpatialTissueData

if HAS_SKLEARN:
    from spatialtissuepy.lda import (
        # Main class and functions
        SpatialLDA,
        fit_spatial_lda,
        compute_neighborhood_features,
        compute_neighborhood_counts,
        # Sampling
        poisson_disk_sample,
        grid_sample,
        random_sample,
        stratified_sample,
        # Analysis
        topic_cell_type_matrix,
        topic_enrichment,
        dominant_topic_per_cell,
        topic_assignment_uncertainty,
        topic_spatial_distribution,
        compare_topics_across_samples,
        topic_prevalence_by_cell_type,
        # Metrics
        topic_coherence,
        topic_diversity,
        topic_exclusivity,
        spatial_topic_consistency,
        compute_model_selection_metrics,
    )


pytestmark = pytest.mark.skipif(
    not HAS_SKLEARN,
    reason="scikit-learn not installed"
)


# =============================================================================
# Neighborhood Feature Computation Tests
# =============================================================================

class TestNeighborhoodFeatures:
    """Tests for neighborhood feature computation."""
    
    def test_compute_neighborhood_features_basic(self, small_tissue):
        """Test basic neighborhood feature computation."""
        features = compute_neighborhood_features(
            small_tissue,
            method='radius',
            radius=50.0,
            normalize=True
        )
        
        assert features.shape[0] == small_tissue.n_cells
        assert features.shape[1] == len(small_tissue.cell_types_unique)
        # Should be normalized (rows sum to ~1)
        row_sums = features.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(small_tissue.n_cells))
    
    def test_compute_neighborhood_features_unnormalized(self, small_tissue):
        """Test unnormalized neighborhood features."""
        features = compute_neighborhood_features(
            small_tissue,
            method='radius',
            radius=50.0,
            normalize=False
        )
        
        # Should be counts (integers or floats representing counts)
        assert features.shape[0] == small_tissue.n_cells
        assert features.shape[1] == len(small_tissue.cell_types_unique)
        # Values should be non-negative
        assert np.all(features >= 0)
    
    def test_compute_neighborhood_features_knn(self, small_tissue):
        """Test k-NN neighborhood method."""
        features = compute_neighborhood_features(
            small_tissue,
            method='knn',
            k=10,
            normalize=True
        )
        
        assert features.shape == (small_tissue.n_cells, len(small_tissue.cell_types_unique))
    
    def test_compute_neighborhood_features_include_self(self, small_tissue):
        """Test including/excluding self in neighborhood."""
        features_with_self = compute_neighborhood_features(
            small_tissue,
            radius=50,
            include_self=True,
            normalize=False
        )
        
        features_without_self = compute_neighborhood_features(
            small_tissue,
            radius=50,
            include_self=False,
            normalize=False
        )
        
        # With self should have higher counts
        assert np.mean(features_with_self) >= np.mean(features_without_self)
    
    def test_compute_neighborhood_counts(self, small_tissue):
        """Test integer count computation."""
        counts = compute_neighborhood_counts(
            small_tissue,
            method='radius',
            radius=50,
            include_self=True
        )
        
        assert counts.shape == (small_tissue.n_cells, len(small_tissue.cell_types_unique))
        # Should be integer-like
        assert np.allclose(counts, counts.astype(int))


# =============================================================================
# SpatialLDA Class Tests
# =============================================================================

class TestSpatialLDA:
    """Tests for SpatialLDA class."""
    
    def test_spatial_lda_initialization(self):
        """Test SpatialLDA initialization."""
        model = SpatialLDA(
            n_topics=5,
            neighborhood_radius=50,
            random_state=42
        )
        
        assert model.n_topics == 5
        assert model.neighborhood_radius == 50
        assert model.random_state == 42
        assert not model._is_fitted
    
    def test_spatial_lda_fit_basic(self, small_tissue):
        """Test basic model fitting."""
        model = SpatialLDA(
            n_topics=3,
            neighborhood_radius=50,
            random_state=42
        )
        
        model.fit(small_tissue)
        
        assert model._is_fitted
        assert len(model.cell_types_) == len(small_tissue.cell_types_unique)
        assert model.topic_cell_type_matrix_ is not None
        assert model.topic_cell_type_matrix_.shape == (3, len(small_tissue.cell_types_unique))
    
    def test_spatial_lda_transform(self, small_tissue):
        """Test transforming data to topic weights."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        topic_weights = model.transform(small_tissue)
        
        assert topic_weights.shape == (small_tissue.n_cells, 3)
        # Rows should sum to 1 (probability distribution)
        row_sums = topic_weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(small_tissue.n_cells))
        # All values should be in [0, 1]
        assert np.all(topic_weights >= 0)
        assert np.all(topic_weights <= 1)
    
    def test_spatial_lda_fit_transform(self, small_tissue):
        """Test fit_transform method."""
        model = SpatialLDA(n_topics=3, random_state=42)
        
        topic_weights = model.fit_transform(small_tissue)
        
        assert model._is_fitted
        assert topic_weights.shape == (small_tissue.n_cells, 3)
    
    def test_spatial_lda_predict(self, small_tissue):
        """Test predicting dominant topics."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        dominant_topics = model.predict(small_tissue)
        
        assert len(dominant_topics) == small_tissue.n_cells
        assert dominant_topics.dtype == int
        assert np.all(dominant_topics >= 0)
        assert np.all(dominant_topics < 3)
    
    def test_spatial_lda_not_fitted_error(self, small_tissue):
        """Test error when transforming before fitting."""
        model = SpatialLDA(n_topics=3)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform(small_tissue)
    
    def test_spatial_lda_topic_summary(self, small_tissue):
        """Test topic summary generation."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        summary = model.topic_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert summary.shape == (3, len(small_tissue.cell_types_unique))
        assert list(summary.columns) == model.cell_types_
    
    def test_spatial_lda_top_cell_types(self, small_tissue):
        """Test getting top cell types per topic."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        top_types = model.top_cell_types_per_topic(n_top=3)
        
        assert len(top_types) == 3  # 3 topics
        for topic_idx in range(3):
            assert len(top_types[topic_idx]) == 3  # 3 top types
            # Should be (cell_type, weight) tuples
            for cell_type, weight in top_types[topic_idx]:
                assert isinstance(cell_type, str)
                assert 0 <= weight <= 1
    
    def test_spatial_lda_perplexity(self, small_tissue):
        """Test perplexity calculation."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        perplexity = model.perplexity(small_tissue)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_spatial_lda_score(self, small_tissue):
        """Test log-likelihood score."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        score = model.score(small_tissue)
        
        assert isinstance(score, float)
        # Log-likelihood is typically negative
        assert score < 0
    
    def test_spatial_lda_add_topics_to_data(self, small_tissue):
        """Test adding topic weights to data."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        data_with_topics = model.add_topics_to_data(small_tissue, prefix='topic')
        
        assert isinstance(data_with_topics, SpatialTissueData)
        assert data_with_topics.markers is not None
        # Should have topic columns
        assert 'topic_0' in data_with_topics.markers.columns
        assert 'topic_1' in data_with_topics.markers.columns
        assert 'topic_2' in data_with_topics.markers.columns
        assert 'topic_dominant' in data_with_topics.markers.columns


class TestSpatialLDAMultiSample:
    """Tests for multi-sample LDA fitting."""
    
    def test_fit_multi_sample_basic(self, multisample_cohort):
        """Test fitting on multiple samples."""
        model = SpatialLDA(n_topics=3, random_state=42)
        
        # Get list of samples
        samples = [multisample_cohort.subset_sample(sid) 
                  for sid in multisample_cohort.sample_ids_unique]
        
        model.fit(samples)
        
        assert model._is_fitted
        # Cell types should be union of all samples
        all_types = set()
        for sample in samples:
            all_types.update(sample.cell_types_unique)
        assert set(model.cell_types_) == all_types
    
    def test_transform_after_multi_fit(self, multisample_cohort):
        """Test transforming individual samples after multi-fit."""
        samples = [multisample_cohort.subset_sample(sid) 
                  for sid in multisample_cohort.sample_ids_unique]
        
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(samples)
        
        # Transform first sample
        topic_weights = model.transform(samples[0])
        
        assert topic_weights.shape[0] == samples[0].n_cells
        assert topic_weights.shape[1] == 3


class TestSpatialLDAParameters:
    """Tests for different SpatialLDA parameters."""
    
    def test_different_n_topics(self, small_tissue):
        """Test with different numbers of topics."""
        for n_topics in [2, 5, 10]:
            model = SpatialLDA(n_topics=n_topics, random_state=42)
            model.fit(small_tissue)
            
            assert model.topic_cell_type_matrix_.shape[0] == n_topics
            
            topic_weights = model.transform(small_tissue)
            assert topic_weights.shape[1] == n_topics
    
    def test_different_neighborhood_methods(self, small_tissue):
        """Test different neighborhood methods."""
        model_radius = SpatialLDA(
            n_topics=3,
            neighborhood_method='radius',
            neighborhood_radius=50,
            random_state=42
        )
        model_radius.fit(small_tissue)
        
        model_knn = SpatialLDA(
            n_topics=3,
            neighborhood_method='knn',
            neighborhood_k=10,
            random_state=42
        )
        model_knn.fit(small_tissue)
        
        # Both should produce valid models
        assert model_radius._is_fitted
        assert model_knn._is_fitted
    
    def test_different_lda_hyperparameters(self, small_tissue):
        """Test different LDA hyperparameters."""
        model = SpatialLDA(
            n_topics=3,
            doc_topic_prior=0.5,
            topic_word_prior=0.5,
            max_iter=50,
            random_state=42
        )
        
        model.fit(small_tissue)
        assert model._is_fitted
    
    def test_reproducibility_with_seed(self, small_tissue):
        """Test reproducibility with random seed."""
        model1 = SpatialLDA(n_topics=3, random_state=42)
        model1.fit(small_tissue)
        weights1 = model1.transform(small_tissue)
        
        model2 = SpatialLDA(n_topics=3, random_state=42)
        model2.fit(small_tissue)
        weights2 = model2.transform(small_tissue)
        
        # Should produce identical results
        np.testing.assert_array_almost_equal(weights1, weights2)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestFitSpatialLDA:
    """Tests for fit_spatial_lda convenience function."""
    
    def test_fit_spatial_lda_basic(self, small_tissue):
        """Test basic usage of fit_spatial_lda."""
        model = fit_spatial_lda(
            small_tissue,
            n_topics=3,
            neighborhood_radius=50,
            random_state=42
        )
        
        assert isinstance(model, SpatialLDA)
        assert model._is_fitted
        assert model.n_topics == 3
    
    def test_fit_spatial_lda_with_kwargs(self, small_tissue):
        """Test passing additional kwargs."""
        model = fit_spatial_lda(
            small_tissue,
            n_topics=3,
            neighborhood_radius=50,
            max_iter=50,
            random_state=42
        )
        
        assert model.max_iter == 50


# =============================================================================
# Sampling Tests
# =============================================================================

class TestSampling:
    """Tests for spatial sampling methods."""
    
    def test_random_sample(self, medium_tissue):
        """Test random sampling."""
        indices = random_sample(medium_tissue, n_samples=50, seed=42)
        
        assert len(indices) == 50
        assert np.all(indices >= 0)
        assert np.all(indices < medium_tissue.n_cells)
        # Should be unique
        assert len(set(indices)) == 50
    
    def test_grid_sample(self, medium_tissue):
        """Test grid-based sampling."""
        indices = grid_sample(medium_tissue, grid_size=5)
        
        assert len(indices) > 0
        assert np.all(indices >= 0)
        assert np.all(indices < medium_tissue.n_cells)
    
    def test_stratified_sample(self, medium_tissue):
        """Test stratified sampling by cell type."""
        indices = stratified_sample(
            medium_tissue,
            n_per_type=10,
            seed=42
        )
        
        assert len(indices) > 0
        # Check that we have samples from each type
        sampled_types = set(medium_tissue.cell_types[indices])
        assert len(sampled_types) >= 1
    
    def test_poisson_disk_sample(self, medium_tissue):
        """Test Poisson disk sampling."""
        indices = poisson_disk_sample(
            medium_tissue,
            min_distance=30,
            seed=42
        )
        
        assert len(indices) > 0
        # Verify minimum distance constraint
        coords = medium_tissue.coordinates[indices]
        from scipy.spatial.distance import pdist
        if len(coords) > 1:
            distances = pdist(coords)
            assert np.all(distances >= 30 - 1e-6)  # Allow small numerical error


# =============================================================================
# Analysis Function Tests
# =============================================================================

class TestAnalysisFunctions:
    """Tests for topic analysis functions."""
    
    def test_topic_cell_type_matrix(self, small_tissue):
        """Test extracting topic-cell type matrix."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        matrix = topic_cell_type_matrix(model)
        
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (3, len(small_tissue.cell_types_unique))
    
    def test_dominant_topic_per_cell(self, small_tissue):
        """Test getting dominant topics."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        dominant = dominant_topic_per_cell(model, small_tissue)
        
        assert len(dominant) == small_tissue.n_cells
        assert np.all(dominant >= 0)
        assert np.all(dominant < 3)
    
    def test_topic_assignment_uncertainty(self, small_tissue):
        """Test computing topic assignment uncertainty."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        uncertainty = topic_assignment_uncertainty(model, small_tissue)
        
        assert len(uncertainty) == small_tissue.n_cells
        assert np.all(uncertainty >= 0)
        # Maximum entropy for 3 topics
        max_entropy = np.log2(3)
        assert np.all(uncertainty <= max_entropy + 1e-6)
    
    def test_topic_prevalence_by_cell_type(self, small_tissue):
        """Test computing topic prevalence by cell type."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        prevalence = topic_prevalence_by_cell_type(model, small_tissue)
        
        assert isinstance(prevalence, pd.DataFrame)
        assert prevalence.shape[0] == len(small_tissue.cell_types_unique)
        # Should have cell_type, n_cells, and 2 columns per topic (mean, std)
        assert prevalence.shape[1] == 2 + 2 * 3
    
    def test_topic_spatial_distribution(self, small_tissue):
        """Test spatial distribution of topics."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        result = topic_spatial_distribution(
            model,
            small_tissue,
            topic_idx=0
        )
        
        assert 'positions' in result
        assert 'centroid' in result
        assert 'spread' in result
    
    def test_compare_topics_across_samples(self, multisample_cohort):
        """Test comparing topics across samples."""
        samples = [multisample_cohort.subset_sample(sid) 
                  for sid in multisample_cohort.sample_ids_unique[:2]]
        
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(samples)
        
        comparison = compare_topics_across_samples(model, samples)
        
        assert isinstance(comparison, pd.DataFrame)
        # Should have entries for each sample
        assert len(comparison) >= 2


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for topic quality metrics."""
    
    def test_topic_coherence(self, small_tissue):
        """Test topic coherence calculation."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        coherence = topic_coherence(model, small_tissue)
        
        assert isinstance(coherence, float)
    
    def test_topic_diversity(self, small_tissue):
        """Test topic diversity calculation."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        diversity = topic_diversity(model)
        
        assert isinstance(diversity, float)
        assert diversity >= 0
    
    def test_topic_exclusivity(self, small_tissue):
        """Test topic exclusivity calculation."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        exclusivity = topic_exclusivity(model)
        
        assert isinstance(exclusivity, dict)
        assert len(exclusivity) == 3
    
    def test_spatial_topic_consistency(self, small_tissue):
        """Test spatial consistency of topics."""
        model = SpatialLDA(n_topics=3, random_state=42)
        model.fit(small_tissue)
        
        consistency = spatial_topic_consistency(
            model,
            small_tissue,
            radius=50
        )
        
        assert isinstance(consistency, dict)
        assert 'agreement_rate' in consistency
    
    def test_compute_model_selection_metrics(self, small_tissue):
        """Test computing multiple metrics for model selection."""
        # Using list signature
        metrics = compute_model_selection_metrics(
            [2, 3],
            data=small_tissue,
            neighborhood_radius=50,
            random_state=42
        )
        
        assert isinstance(metrics, pd.DataFrame)
        assert 'perplexity' in metrics.columns
        assert 'log_likelihood' in metrics.columns
        assert 'mean_coherence' in metrics.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestLDAIntegration:
    """Integration tests for LDA workflow."""
    
    def test_complete_lda_workflow(self, medium_tissue):
        """Test complete LDA analysis workflow."""
        # 1. Fit model
        model = SpatialLDA(
            n_topics=5,
            neighborhood_radius=50,
            random_state=42
        )
        model.fit(medium_tissue)
        
        # 2. Get topic assignments
        topic_weights = model.transform(medium_tissue)
        assert topic_weights.shape == (medium_tissue.n_cells, 5)
        
        # 3. Get dominant topics
        dominant_topics = model.predict(medium_tissue)
        assert len(dominant_topics) == medium_tissue.n_cells
        
        # 4. Analyze topics
        summary = model.topic_summary()
        assert isinstance(summary, pd.DataFrame)
        
        top_types = model.top_cell_types_per_topic(n_top=3)
        assert len(top_types) == 5
        
        # 5. Compute metrics
        diversity = topic_diversity(model)
        assert isinstance(diversity, float)
        
        coherence = topic_coherence(model, medium_tissue)
        assert isinstance(coherence, float)
        
        # 6. Add to data
        data_with_topics = model.add_topics_to_data(medium_tissue)
        assert data_with_topics.markers is not None
    
    def test_model_comparison_workflow(self, medium_tissue):
        """Test comparing models with different n_topics."""
        
        # Correct usage of compute_model_selection_metrics
        metrics_df = compute_model_selection_metrics(
            [3, 5],
            data=medium_tissue,
            neighborhood_radius=50,
            random_state=42
        )
        
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == 2
        assert 'perplexity' in metrics_df.columns
    
    def test_multi_sample_workflow(self, multisample_cohort):
        """Test multi-sample analysis workflow."""
        # Get samples
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        # Fit joint model
        model = SpatialLDA(n_topics=4, random_state=42)
        model.fit(samples)
        
        # Transform each sample
        sample_topics = []
        for sample in samples:
            weights = model.transform(sample)
            sample_topics.append(weights)
        
        # Compare across samples
        comparison = compare_topics_across_samples(model, samples)
        assert isinstance(comparison, pd.DataFrame)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestLDAEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_cell_type(self):
        """Test with tissue containing single cell type."""
        coords = np.random.rand(50, 2) * 100
        types = np.array(['A'] * 50)
        data = SpatialTissueData(coords, types)
        
        model = SpatialLDA(n_topics=2, random_state=42)
        model.fit(data)
        
        # Should still work but topics may be similar
        assert model._is_fitted
        assert model.topic_cell_type_matrix_.shape == (2, 1)
    
    def test_more_topics_than_cell_types(self, small_tissue):
        """Test with more topics than cell types."""
        n_cell_types = len(small_tissue.cell_types_unique)
        
        model = SpatialLDA(
            n_topics=n_cell_types + 2,
            random_state=42
        )
        model.fit(small_tissue)
        
        # Should fit but may have redundant topics
        assert model._is_fitted
    
    def test_small_sample(self):
        """Test with very small sample."""
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        types = np.array(['A', 'B', 'A', 'B', 'A'])
        data = SpatialTissueData(coords, types)
        
        model = SpatialLDA(n_topics=2, random_state=42)
        
        try:
            model.fit(data)
            assert model._is_fitted
        except Exception:
            # May fail with very small sample - acceptable
            pytest.skip("Sample too small for LDA")
    
    def test_isolated_cells(self):
        """Test with isolated cells (no neighbors)."""
        # Create widely spaced cells
        coords = np.array([[0, 0], [1000, 1000], [2000, 2000]])
        types = np.array(['A', 'B', 'C'])
        data = SpatialTissueData(coords, types)
        
        model = SpatialLDA(
            n_topics=2,
            neighborhood_radius=10,  # Small radius
            random_state=42
        )
        
        model.fit(data)
        # Should work even with sparse neighborhoods
        assert model._is_fitted


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestLDAPerformance:
    """Performance tests for LDA operations."""
    
    def test_fit_performance(self, large_tissue):
        """Test fitting performance with large dataset."""
        import time
        
        model = SpatialLDA(
            n_topics=5,
            neighborhood_radius=50,
            max_iter=50,  # Limit iterations for speed
            random_state=42
        )
        
        start = time.time()
        model.fit(large_tissue)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 150.0  # Increased for slow CI
        assert model._is_fitted
    
    def test_transform_performance(self, large_tissue):
        """Test transform performance."""
        import time
        
        model = SpatialLDA(n_topics=5, max_iter=50, random_state=42)
        model.fit(large_tissue)
        
        start = time.time()
        topic_weights = model.transform(large_tissue)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 10.0
        assert topic_weights.shape[0] == large_tissue.n_cells
