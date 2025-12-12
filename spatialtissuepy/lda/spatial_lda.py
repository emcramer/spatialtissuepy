"""
Core Spatial LDA implementation.

This module provides the main SpatialLDA class and supporting functions for
fitting topic models to spatial tissue data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Any, Union, Tuple,
    TYPE_CHECKING
)
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Neighborhood Feature Computation
# -----------------------------------------------------------------------------

def compute_neighborhood_features(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: float = 50.0,
    k: int = 30,
    include_self: bool = True,
    normalize: bool = True,
    pseudocount: float = 0.0
) -> np.ndarray:
    """
    Compute neighborhood composition features for each cell.
    
    This creates a "document" representation of each cell's neighborhood,
    where the "words" are the cell types present in the neighborhood.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data with coordinates and cell types.
    method : str, default 'radius'
        Neighborhood method: 'radius' or 'knn'.
    radius : float, default 50.0
        Radius for 'radius' method (in spatial units).
    k : int, default 30
        Number of neighbors for 'knn' method.
    include_self : bool, default True
        Whether to include the focal cell in its neighborhood.
    normalize : bool, default True
        If True, normalize to proportions (rows sum to 1).
        If False, return raw counts.
    pseudocount : float, default 0.0
        Pseudocount to add for smoothing (useful for sparse neighborhoods).
        
    Returns
    -------
    np.ndarray
        Neighborhood composition matrix of shape (n_cells, n_cell_types).
        Each row represents a cell's neighborhood composition.
        
    Notes
    -----
    This is equivalent to treating each cell's neighborhood as a "document"
    in the LDA framework, where the cell types are the "vocabulary".
    
    Examples
    --------
    >>> features = compute_neighborhood_features(data, method='radius', radius=50)
    >>> print(f"Shape: {features.shape}")  # (n_cells, n_cell_types)
    """
    # Use the existing neighborhood_composition function from spatial module
    from spatialtissuepy.spatial.neighborhood import neighborhood_composition
    
    return neighborhood_composition(
        data,
        method=method,
        radius=radius,
        k=k,
        include_self=include_self,
        normalize=normalize,
        pseudocount=pseudocount
    )


def compute_neighborhood_counts(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: float = 50.0,
    k: int = 30,
    include_self: bool = True
) -> np.ndarray:
    """
    Compute raw neighborhood counts (integer counts for LDA).
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Neighborhood method.
    radius : float, default 50.0
        Radius for 'radius' method.
    k : int, default 30
        Number of neighbors for 'knn' method.
    include_self : bool, default True
        Whether to include focal cell.
        
    Returns
    -------
    np.ndarray
        Integer count matrix of shape (n_cells, n_cell_types).
    """
    from spatialtissuepy.spatial.neighborhood import neighborhood_counts
    
    return neighborhood_counts(
        data,
        method=method,
        radius=radius,
        k=k,
        include_self=include_self
    )


# -----------------------------------------------------------------------------
# Spatial LDA Class
# -----------------------------------------------------------------------------

@dataclass
class SpatialLDA:
    """
    Spatial Latent Dirichlet Allocation for cellular neighborhood analysis.
    
    This class wraps scikit-learn's LDA implementation with spatial tissue-specific
    functionality for discovering recurrent cellular microenvironment patterns.
    
    Attributes
    ----------
    n_topics : int
        Number of topics (microenvironment types) to discover.
    neighborhood_method : str
        Method for computing neighborhoods ('radius' or 'knn').
    neighborhood_radius : float
        Radius for radius-based neighborhoods.
    neighborhood_k : int
        Number of neighbors for KNN-based neighborhoods.
    cell_types : list of str
        Names of cell types (vocabulary).
    topic_cell_type_matrix_ : np.ndarray
        Fitted topic-cell type matrix (n_topics, n_cell_types).
    
    Examples
    --------
    >>> from spatialtissuepy.lda import SpatialLDA
    >>> 
    >>> # Create and fit model
    >>> slda = SpatialLDA(n_topics=5, neighborhood_radius=50)
    >>> slda.fit(data)
    >>> 
    >>> # Get topic assignments for cells
    >>> topic_weights = slda.transform(data)
    >>> dominant_topics = slda.predict(data)
    >>> 
    >>> # Analyze topics
    >>> print(slda.topic_summary())
    """
    n_topics: int = 5
    neighborhood_method: str = 'radius'
    neighborhood_radius: float = 50.0
    neighborhood_k: int = 30
    include_self: bool = True
    
    # LDA hyperparameters
    doc_topic_prior: Optional[float] = None  # Alpha (Dirichlet prior on topics)
    topic_word_prior: Optional[float] = None  # Beta (Dirichlet prior on words)
    learning_method: str = 'batch'  # 'batch' or 'online'
    max_iter: int = 100
    random_state: Optional[int] = None
    
    # Fitted attributes
    cell_types_: List[str] = field(default_factory=list)
    topic_cell_type_matrix_: Optional[np.ndarray] = field(default=None, repr=False)
    _lda_model: Any = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)
    
    def fit(
        self,
        data: Union['SpatialTissueData', List['SpatialTissueData']],
        sample_indices: Optional[np.ndarray] = None
    ) -> 'SpatialLDA':
        """
        Fit the Spatial LDA model to tissue data.
        
        Parameters
        ----------
        data : SpatialTissueData or list of SpatialTissueData
            Single sample or multiple samples to fit jointly.
        sample_indices : np.ndarray, optional
            If provided, fit only on these cell indices.
            
        Returns
        -------
        self
            Fitted model.
        """
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Handle multiple samples
        if isinstance(data, list):
            return self._fit_multi_sample(data)
        
        # Store cell types
        self.cell_types_ = list(data.cell_types_unique)
        
        # Compute neighborhood features
        features = compute_neighborhood_counts(
            data,
            method=self.neighborhood_method,
            radius=self.neighborhood_radius,
            k=self.neighborhood_k,
            include_self=self.include_self
        )
        
        # Subset if indices provided
        if sample_indices is not None:
            features = features[sample_indices]
        
        # Create and fit LDA model
        self._lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            learning_method=self.learning_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        
        self._lda_model.fit(features)
        
        # Store topic-cell type matrix (normalized)
        self.topic_cell_type_matrix_ = self._lda_model.components_
        # Normalize rows to sum to 1
        row_sums = self.topic_cell_type_matrix_.sum(axis=1, keepdims=True)
        self.topic_cell_type_matrix_ = self.topic_cell_type_matrix_ / row_sums
        
        self._is_fitted = True
        
        return self
    
    def _fit_multi_sample(
        self,
        samples: List['SpatialTissueData']
    ) -> 'SpatialLDA':
        """Fit model on multiple samples jointly."""
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Get union of cell types
        all_cell_types = set()
        for sample in samples:
            all_cell_types.update(sample.cell_types_unique)
        self.cell_types_ = sorted(list(all_cell_types))
        
        # Compute features for all samples
        all_features = []
        
        for sample in samples:
            # Create mapping for this sample's cell types
            sample_types = list(sample.cell_types_unique)
            type_to_idx = {ct: i for i, ct in enumerate(self.cell_types_)}
            
            # Compute neighborhood counts
            counts = compute_neighborhood_counts(
                sample,
                method=self.neighborhood_method,
                radius=self.neighborhood_radius,
                k=self.neighborhood_k,
                include_self=self.include_self
            )
            
            # Remap to unified cell type ordering
            n_cells = counts.shape[0]
            unified_counts = np.zeros((n_cells, len(self.cell_types_)))
            
            for i, ct in enumerate(sample_types):
                unified_idx = type_to_idx[ct]
                unified_counts[:, unified_idx] = counts[:, i]
            
            all_features.append(unified_counts)
        
        # Concatenate all features
        features = np.vstack(all_features)
        
        # Fit LDA
        self._lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            learning_method=self.learning_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        
        self._lda_model.fit(features)
        
        # Store normalized topic-cell type matrix
        self.topic_cell_type_matrix_ = self._lda_model.components_
        row_sums = self.topic_cell_type_matrix_.sum(axis=1, keepdims=True)
        self.topic_cell_type_matrix_ = self.topic_cell_type_matrix_ / row_sums
        
        self._is_fitted = True
        
        return self
    
    def transform(
        self,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        """
        Get topic weights for each cell in the data.
        
        Parameters
        ----------
        data : SpatialTissueData
            Spatial tissue data.
            
        Returns
        -------
        np.ndarray
            Topic weight matrix of shape (n_cells, n_topics).
            Each row sums to 1.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Compute neighborhood features
        features = self._prepare_features(data)
        
        # Transform using LDA
        topic_weights = self._lda_model.transform(features)
        
        return topic_weights
    
    def fit_transform(
        self,
        data: Union['SpatialTissueData', List['SpatialTissueData']]
    ) -> np.ndarray:
        """
        Fit the model and return topic weights.
        
        Parameters
        ----------
        data : SpatialTissueData or list
            Data to fit.
            
        Returns
        -------
        np.ndarray
            Topic weights for the fitted data.
        """
        self.fit(data)
        
        if isinstance(data, list):
            # Return weights for first sample
            return self.transform(data[0])
        return self.transform(data)
    
    def predict(
        self,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        """
        Get dominant topic assignment for each cell.
        
        Parameters
        ----------
        data : SpatialTissueData
            Spatial tissue data.
            
        Returns
        -------
        np.ndarray
            Integer array of dominant topic indices (0 to n_topics-1).
        """
        topic_weights = self.transform(data)
        return np.argmax(topic_weights, axis=1)
    
    def _prepare_features(
        self,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        """Prepare features with consistent cell type ordering."""
        sample_types = list(data.cell_types_unique)
        
        # Compute counts
        counts = compute_neighborhood_counts(
            data,
            method=self.neighborhood_method,
            radius=self.neighborhood_radius,
            k=self.neighborhood_k,
            include_self=self.include_self
        )
        
        # Check if cell types match
        if set(sample_types) == set(self.cell_types_) and \
           list(sample_types) == list(self.cell_types_):
            return counts
        
        # Remap to fitted cell type ordering
        type_to_idx = {ct: i for i, ct in enumerate(self.cell_types_)}
        n_cells = counts.shape[0]
        unified_counts = np.zeros((n_cells, len(self.cell_types_)))
        
        for i, ct in enumerate(sample_types):
            if ct in type_to_idx:
                unified_idx = type_to_idx[ct]
                unified_counts[:, unified_idx] = counts[:, i]
        
        return unified_counts
    
    def topic_summary(self) -> pd.DataFrame:
        """
        Get a summary of topic-cell type associations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topics as rows and cell types as columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        return pd.DataFrame(
            self.topic_cell_type_matrix_,
            index=[f'Topic_{i}' for i in range(self.n_topics)],
            columns=self.cell_types_
        )
    
    def top_cell_types_per_topic(
        self,
        n_top: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the top cell types for each topic.
        
        Parameters
        ----------
        n_top : int, default 5
            Number of top cell types to return per topic.
            
        Returns
        -------
        dict
            Mapping from topic index to list of (cell_type, weight) tuples.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        result = {}
        
        for topic_idx in range(self.n_topics):
            weights = self.topic_cell_type_matrix_[topic_idx]
            top_indices = np.argsort(weights)[::-1][:n_top]
            
            result[topic_idx] = [
                (self.cell_types_[i], float(weights[i]))
                for i in top_indices
            ]
        
        return result
    
    def perplexity(
        self,
        data: 'SpatialTissueData'
    ) -> float:
        """
        Compute perplexity on held-out data.
        
        Lower perplexity indicates better fit.
        
        Parameters
        ----------
        data : SpatialTissueData
            Data to evaluate.
            
        Returns
        -------
        float
            Perplexity score.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        features = self._prepare_features(data)
        return self._lda_model.perplexity(features)
    
    def score(
        self,
        data: 'SpatialTissueData'
    ) -> float:
        """
        Compute log-likelihood on data.
        
        Higher is better.
        
        Parameters
        ----------
        data : SpatialTissueData
            Data to evaluate.
            
        Returns
        -------
        float
            Log-likelihood score.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        features = self._prepare_features(data)
        return self._lda_model.score(features)
    
    def add_topics_to_data(
        self,
        data: 'SpatialTissueData',
        prefix: str = 'topic'
    ) -> 'SpatialTissueData':
        """
        Add topic weights as custom data to SpatialTissueData.
        
        Parameters
        ----------
        data : SpatialTissueData
            Data to annotate.
        prefix : str, default 'topic'
            Prefix for topic column names.
            
        Returns
        -------
        SpatialTissueData
            Data with topic weights added to markers/custom data.
        """
        topic_weights = self.transform(data)
        
        # Create DataFrame with topic weights
        topic_df = pd.DataFrame(
            topic_weights,
            columns=[f'{prefix}_{i}' for i in range(self.n_topics)]
        )
        
        # Add dominant topic
        topic_df[f'{prefix}_dominant'] = np.argmax(topic_weights, axis=1)
        
        # Merge with existing markers
        if data.markers is not None:
            new_markers = pd.concat([data.markers.reset_index(drop=True), topic_df], axis=1)
        else:
            new_markers = topic_df
        
        # Create new data object
        from spatialtissuepy.core import SpatialTissueData
        
        return SpatialTissueData(
            coordinates=data._coordinates.copy(),
            cell_types=data._cell_types.copy(),
            markers=new_markers,
            metadata=data.metadata.copy()
        )


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def fit_spatial_lda(
    data: Union['SpatialTissueData', List['SpatialTissueData']],
    n_topics: int = 5,
    neighborhood_radius: float = 50.0,
    neighborhood_method: str = 'radius',
    **kwargs
) -> SpatialLDA:
    """
    Fit a Spatial LDA model to tissue data.
    
    Convenience function for quick model fitting.
    
    Parameters
    ----------
    data : SpatialTissueData or list
        Data to fit.
    n_topics : int, default 5
        Number of topics to discover.
    neighborhood_radius : float, default 50.0
        Radius for neighborhood computation.
    neighborhood_method : str, default 'radius'
        Method for computing neighborhoods.
    **kwargs
        Additional arguments passed to SpatialLDA.
        
    Returns
    -------
    SpatialLDA
        Fitted model.
        
    Examples
    --------
    >>> model = fit_spatial_lda(data, n_topics=8, neighborhood_radius=30)
    >>> topic_weights = model.transform(data)
    """
    model = SpatialLDA(
        n_topics=n_topics,
        neighborhood_radius=neighborhood_radius,
        neighborhood_method=neighborhood_method,
        **kwargs
    )
    
    model.fit(data)
    
    return model
