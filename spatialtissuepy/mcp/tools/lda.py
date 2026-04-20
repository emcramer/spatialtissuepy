"""
LDA tools for MCP server.

Spatial topic modeling tools.

Tools (8 total):
- lda_fit: Fit LDA model
- lda_transform: Get topic weights
- lda_get_topic_composition: Topic-cell type matrix
- lda_get_dominant_topics: Assign cells to topics
- lda_topic_coherence: Compute coherence
- lda_topic_diversity: Compute diversity
- lda_topic_spatial_consistency: Spatial consistency
- lda_select_n_topics: Model selection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


class LDAFitResult(BaseModel):
    """Result of fitting an LDA model."""

    session_id: str
    data_key: str
    model_key: str
    n_topics: int
    neighborhood_radius: float
    n_cells: int
    cell_types: List[str]
    perplexity: Optional[float] = None


class TopicWeightsResult(BaseModel):
    """Result of topic weight transformation."""

    session_id: str
    model_key: str
    n_cells: int
    n_topics: int
    dominant_topic_counts: Dict[int, int] = Field(description="Cells per dominant topic")


class TopicCompositionResult(BaseModel):
    """Topic composition matrix."""

    session_id: str
    model_key: str
    n_topics: int
    cell_types: List[str]
    matrix: List[List[float]] = Field(description="Topics x cell_types matrix")


class DominantTopicsResult(BaseModel):
    """Dominant topic assignments."""

    session_id: str
    model_key: str
    n_cells: int
    n_topics: int
    topic_counts: Dict[int, int]
    topic_fractions: Dict[int, float]


class CoherenceResult(BaseModel):
    """Topic coherence result."""

    session_id: str
    model_key: str
    coherence: float
    method: str


class DiversityResult(BaseModel):
    """Topic diversity result."""

    session_id: str
    model_key: str
    diversity: float


class SpatialConsistencyResult(BaseModel):
    """Spatial consistency result."""

    session_id: str
    model_key: str
    consistency: float
    per_topic: List[float]


class ModelSelectionResult(BaseModel):
    """Model selection result."""

    session_id: str
    data_key: str
    n_topics_tested: List[int]
    coherence_scores: List[float]
    best_n_topics: int
    best_coherence: float


def register_tools(mcp: "FastMCP") -> None:
    """Register LDA tools with the MCP server."""

    @mcp.tool()
    def lda_fit(
        session_id: str,
        n_topics: int = 5,
        neighborhood_radius: float = 50.0,
        data_key: str = "primary",
        model_key: str = "lda_model",
        random_state: Optional[int] = 42,
    ) -> LDAFitResult:
        """
        Fit a Spatial LDA model to discover cellular neighborhoods.

        Spatial LDA treats each cell's neighborhood as a "document" and
        cell types as "words", discovering latent topics that represent
        recurring spatial patterns.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        n_topics : int
            Number of topics to discover.
        neighborhood_radius : float
            Radius for defining cell neighborhoods.
        data_key : str
            Key of the spatial data.
        model_key : str
            Key to store the fitted model.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        LDAFitResult
            Information about the fitted model.
        """
        from spatialtissuepy.lda import SpatialLDA
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        model = SpatialLDA(
            n_topics=n_topics,
            neighborhood_radius=neighborhood_radius,
            random_state=random_state,
        )
        model.fit(data)

        session_mgr.store_model(session_id, model_key, model, "lda")

        return LDAFitResult(
            session_id=session_id,
            data_key=data_key,
            model_key=model_key,
            n_topics=n_topics,
            neighborhood_radius=neighborhood_radius,
            n_cells=data.n_cells,
            cell_types=list(data.cell_types_unique),
            perplexity=getattr(model, "perplexity_", None),
        )

    @mcp.tool()
    def lda_transform(
        session_id: str,
        model_key: str = "lda_model",
        data_key: str = "primary",
    ) -> TopicWeightsResult:
        """
        Get topic weights for each cell using a fitted LDA model.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the fitted LDA model.
        data_key : str
            Key of the spatial data.

        Returns
        -------
        TopicWeightsResult
            Summary of topic weight distribution.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        weights = model.transform(data)
        dominant = np.argmax(weights, axis=1)

        unique, counts = np.unique(dominant, return_counts=True)
        dominant_counts = {int(t): int(c) for t, c in zip(unique, counts)}

        return TopicWeightsResult(
            session_id=session_id,
            model_key=model_key,
            n_cells=data.n_cells,
            n_topics=weights.shape[1],
            dominant_topic_counts=dominant_counts,
        )

    @mcp.tool()
    def lda_get_topic_composition(
        session_id: str,
        model_key: str = "lda_model",
    ) -> TopicCompositionResult:
        """
        Get topic-cell type composition matrix.

        Shows the cell type distribution for each topic.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the LDA model.

        Returns
        -------
        TopicCompositionResult
            Matrix of topic compositions.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")

        components = model.components_
        cell_types = list(model.cell_types_)

        return TopicCompositionResult(
            session_id=session_id,
            model_key=model_key,
            n_topics=components.shape[0],
            cell_types=cell_types,
            matrix=components.tolist(),
        )

    @mcp.tool()
    def lda_get_dominant_topics(
        session_id: str,
        model_key: str = "lda_model",
        data_key: str = "primary",
    ) -> DominantTopicsResult:
        """
        Get dominant topic assignment for each cell.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the LDA model.
        data_key : str
            Key of the spatial data.

        Returns
        -------
        DominantTopicsResult
            Topic assignment distribution.
        """
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        weights = model.transform(data)
        dominant = np.argmax(weights, axis=1)

        unique, counts = np.unique(dominant, return_counts=True)
        topic_counts = {int(t): int(c) for t, c in zip(unique, counts)}
        topic_fractions = {int(t): float(c) / len(dominant) for t, c in zip(unique, counts)}

        return DominantTopicsResult(
            session_id=session_id,
            model_key=model_key,
            n_cells=data.n_cells,
            n_topics=weights.shape[1],
            topic_counts=topic_counts,
            topic_fractions=topic_fractions,
        )

    @mcp.tool()
    def lda_topic_coherence(
        session_id: str,
        model_key: str = "lda_model",
        data_key: str = "primary",
        method: str = "umass",
    ) -> CoherenceResult:
        """
        Compute topic coherence score.

        Higher coherence indicates more interpretable topics.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the LDA model.
        data_key : str
            Key of the spatial data.
        method : str
            Coherence method: "umass" or "cv".

        Returns
        -------
        CoherenceResult
            Coherence score.
        """
        from spatialtissuepy.lda import topic_coherence
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        coherence = topic_coherence(model, data, method=method)

        return CoherenceResult(
            session_id=session_id,
            model_key=model_key,
            coherence=float(coherence),
            method=method,
        )

    @mcp.tool()
    def lda_topic_diversity(
        session_id: str,
        model_key: str = "lda_model",
    ) -> DiversityResult:
        """
        Compute topic diversity score.

        Higher diversity means topics are more distinct from each other.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the LDA model.

        Returns
        -------
        DiversityResult
            Diversity score.
        """
        from spatialtissuepy.lda import topic_diversity
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")

        diversity = topic_diversity(model)

        return DiversityResult(
            session_id=session_id,
            model_key=model_key,
            diversity=float(diversity),
        )

    @mcp.tool()
    def lda_topic_spatial_consistency(
        session_id: str,
        model_key: str = "lda_model",
        data_key: str = "primary",
        radius: float = 50.0,
    ) -> SpatialConsistencyResult:
        """
        Compute spatial consistency of topic assignments.

        Measures whether nearby cells have similar topic assignments.

        Parameters
        ----------
        session_id : str
            Session containing the model.
        model_key : str
            Key of the LDA model.
        data_key : str
            Key of the spatial data.
        radius : float
            Neighborhood radius for consistency check.

        Returns
        -------
        SpatialConsistencyResult
            Consistency score overall and per topic.
        """
        from spatialtissuepy.lda import spatial_topic_consistency
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        model = session_mgr.load_model(session_id, model_key)
        data = session_mgr.load_data(session_id, data_key)

        if model is None:
            raise ValueError(f"No model found with key '{model_key}'")
        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        result = spatial_topic_consistency(model, data, radius=radius)

        if isinstance(result, dict):
            overall = result.get("overall", 0)
            per_topic = result.get("per_topic", [])
        else:
            overall = float(result)
            per_topic = []

        return SpatialConsistencyResult(
            session_id=session_id,
            model_key=model_key,
            consistency=float(overall),
            per_topic=[float(x) for x in per_topic],
        )

    @mcp.tool()
    def lda_select_n_topics(
        session_id: str,
        data_key: str = "primary",
        n_topics_range: Optional[List[int]] = None,
        neighborhood_radius: float = 50.0,
        random_state: Optional[int] = 42,
    ) -> ModelSelectionResult:
        """
        Select optimal number of topics using coherence.

        Tests multiple values of n_topics and returns the best.

        Parameters
        ----------
        session_id : str
            Session containing the data.
        data_key : str
            Key of the spatial data.
        n_topics_range : list of int, optional
            Topic counts to test. Default: [3, 5, 7, 10, 15]
        neighborhood_radius : float
            Radius for neighborhood definition.
        random_state : int, optional
            Random seed.

        Returns
        -------
        ModelSelectionResult
            Best n_topics and coherence scores.
        """
        from spatialtissuepy.lda import SpatialLDA, topic_coherence
        from ..server import get_session_manager

        session_mgr = get_session_manager()
        data = session_mgr.load_data(session_id, data_key)

        if data is None:
            raise ValueError(f"No data found with key '{data_key}'")

        if n_topics_range is None:
            n_topics_range = [3, 5, 7, 10, 15]

        coherence_scores = []
        for n in n_topics_range:
            model = SpatialLDA(
                n_topics=n,
                neighborhood_radius=neighborhood_radius,
                random_state=random_state,
            )
            model.fit(data)
            score = topic_coherence(model, data)
            coherence_scores.append(float(score))

        best_idx = int(np.argmax(coherence_scores))
        best_n = n_topics_range[best_idx]
        best_coherence = coherence_scores[best_idx]

        return ModelSelectionResult(
            session_id=session_id,
            data_key=data_key,
            n_topics_tested=n_topics_range,
            coherence_scores=coherence_scores,
            best_n_topics=best_n,
            best_coherence=best_coherence,
        )
