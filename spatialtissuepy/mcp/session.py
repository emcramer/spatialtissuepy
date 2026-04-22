"""
Session management for MCP server.

Provides persistent storage of SpatialTissueData, graphs, and models
across server restarts.

Session Structure
-----------------
~/.spatialtissuepy/mcp_sessions/
└── {session_id}/
    ├── metadata.json        # Session info, timestamps
    ├── data/
    │   └── {key}.pkl        # SpatialTissueData objects
    ├── graphs/
    │   └── {key}.json       # Serialized NetworkX graphs
    └── models/
        └── {key}.json       # LDA/Mapper model parameters
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import uuid
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from spatialtissuepy import SpatialTissueData
    from spatialtissuepy.network import CellGraph
    from spatialtissuepy.lda import SpatialLDA
    from spatialtissuepy.topology import MapperResult


@dataclass
class SessionMetadata:
    """Metadata for a session."""

    session_id: str
    created_at: str
    last_accessed: str
    data_keys: List[str] = field(default_factory=list)
    graph_keys: List[str] = field(default_factory=list)
    model_keys: List[str] = field(default_factory=list)
    panel_keys: List[str] = field(default_factory=list)
    results_cache: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    Manages persistent sessions for MCP server.

    Parameters
    ----------
    base_dir : Path or str
        Base directory for session storage.
        Default: ~/.spatialtissuepy/mcp_sessions/

    Examples
    --------
    >>> manager = SessionManager()
    >>> session_id = manager.create_session()
    >>> manager.store_data(session_id, "primary", spatial_data)
    >>> data = manager.load_data(session_id, "primary")
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            base_dir = Path.home() / ".spatialtissuepy" / "mcp_sessions"
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, SessionMetadata] = {}

    def create_session(self) -> str:
        """
        Create a new session and return its ID.

        Returns
        -------
        str
            Unique session identifier (8 characters).
        """
        session_id = str(uuid.uuid4())[:8]
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        (session_dir / "data").mkdir(exist_ok=True)
        (session_dir / "graphs").mkdir(exist_ok=True)
        (session_dir / "models").mkdir(exist_ok=True)
        (session_dir / "panels").mkdir(exist_ok=True)

        metadata = SessionMetadata(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
        )
        self._save_metadata(session_id, metadata)
        self._active_sessions[session_id] = metadata
        return session_id

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get existing session or create new one.

        Parameters
        ----------
        session_id : str, optional
            Existing session ID to use. If None or not found, creates new.

        Returns
        -------
        str
            Session ID (existing or newly created).
        """
        if session_id and self.session_exists(session_id):
            self._touch_session(session_id)
            return session_id
        return self.create_session()

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return (self.base_dir / session_id / "metadata.json").exists()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with metadata.

        Returns
        -------
        list of dict
            Session information including ID, timestamps, and counts.
        """
        sessions = []
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir():
                meta = self._load_metadata(session_dir.name)
                if meta:
                    sessions.append({
                        "session_id": meta.session_id,
                        "created_at": meta.created_at,
                        "last_accessed": meta.last_accessed,
                        "n_data": len(meta.data_keys),
                        "n_graphs": len(meta.graph_keys),
                        "n_models": len(meta.model_keys),
                        "n_panels": len(meta.panel_keys),
                    })
        return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.

        Parameters
        ----------
        session_id : str
            Session to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            self._active_sessions.pop(session_id, None)
            return True
        return False

    # --- Data Storage (SpatialTissueData) ---

    def store_data(
        self,
        session_id: str,
        key: str,
        data: "SpatialTissueData",
    ) -> None:
        """
        Store SpatialTissueData object.

        Parameters
        ----------
        session_id : str
            Session to store in.
        key : str
            Storage key (e.g., "primary", "subset_tumor").
        data : SpatialTissueData
            Data to store.
        """
        path = self.base_dir / session_id / "data" / f"{key}.pkl"
        self._atomic_pickle_dump(data, path)

        meta = self._load_metadata(session_id)
        if meta and key not in meta.data_keys:
            meta.data_keys.append(key)
            self._save_metadata(session_id, meta)
        self._touch_session(session_id)

    def load_data(
        self,
        session_id: str,
        key: str,
    ) -> Optional["SpatialTissueData"]:
        """
        Load SpatialTissueData object.

        Parameters
        ----------
        session_id : str
            Session to load from.
        key : str
            Storage key.

        Returns
        -------
        SpatialTissueData or None
            Loaded data, or None if not found.
        """
        path = self.base_dir / session_id / "data" / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self._touch_session(session_id)
                return pickle.load(f)
        return None

    def list_data(self, session_id: str) -> List[str]:
        """List stored data keys for a session."""
        meta = self._load_metadata(session_id)
        return meta.data_keys if meta else []

    def delete_data(self, session_id: str, key: str) -> bool:
        """Delete a stored data object."""
        path = self.base_dir / session_id / "data" / f"{key}.pkl"
        if path.exists():
            path.unlink()
            meta = self._load_metadata(session_id)
            if meta and key in meta.data_keys:
                meta.data_keys.remove(key)
                self._save_metadata(session_id, meta)
            return True
        return False

    # --- Graph Storage (NetworkX) ---

    def store_graph(
        self,
        session_id: str,
        key: str,
        graph: "CellGraph",
        params: Optional[Dict] = None,
    ) -> None:
        """
        Store graph with reconstruction info.

        Parameters
        ----------
        session_id : str
            Session to store in.
        key : str
            Storage key.
        graph : CellGraph
            Graph to store.
        params : dict, optional
            Construction parameters for reproducibility.
        """
        from .serialization import serialize_graph

        path = self.base_dir / session_id / "graphs" / f"{key}.json"
        data = serialize_graph(graph, params)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        meta = self._load_metadata(session_id)
        if meta and key not in meta.graph_keys:
            meta.graph_keys.append(key)
            self._save_metadata(session_id, meta)
        self._touch_session(session_id)

    def load_graph(
        self,
        session_id: str,
        key: str,
    ) -> Optional["CellGraph"]:
        """
        Load graph from storage.

        Parameters
        ----------
        session_id : str
            Session to load from.
        key : str
            Storage key.

        Returns
        -------
        CellGraph or None
            Loaded graph, or None if not found.
        """
        from .serialization import deserialize_graph

        path = self.base_dir / session_id / "graphs" / f"{key}.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            self._touch_session(session_id)
            return deserialize_graph(data)
        return None

    def list_graphs(self, session_id: str) -> List[str]:
        """List stored graph keys for a session."""
        meta = self._load_metadata(session_id)
        return meta.graph_keys if meta else []

    # --- Model Storage (LDA, Mapper) ---

    def store_model(
        self,
        session_id: str,
        key: str,
        model: Union["SpatialLDA", "MapperResult"],
        model_type: str,
    ) -> None:
        """
        Store model with parameters for reproducibility.

        Parameters
        ----------
        session_id : str
            Session to store in.
        key : str
            Storage key.
        model : SpatialLDA or MapperResult
            Model to store.
        model_type : str
            Type identifier ("lda" or "mapper").
        """
        from .serialization import serialize_model

        path = self.base_dir / session_id / "models" / f"{key}.json"
        data = serialize_model(model, model_type)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        meta = self._load_metadata(session_id)
        if meta and key not in meta.model_keys:
            meta.model_keys.append(key)
            self._save_metadata(session_id, meta)
        self._touch_session(session_id)

    def load_model(
        self,
        session_id: str,
        key: str,
    ) -> Optional[Union["SpatialLDA", "MapperResult", Dict]]:
        """
        Load model from storage.

        Parameters
        ----------
        session_id : str
            Session to load from.
        key : str
            Storage key.

        Returns
        -------
        SpatialLDA, MapperResult, or dict
            Loaded model, or None if not found.
        """
        from .serialization import deserialize_model

        path = self.base_dir / session_id / "models" / f"{key}.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            self._touch_session(session_id)
            return deserialize_model(data)
        return None

    def list_models(self, session_id: str) -> List[str]:
        """List stored model keys for a session."""
        meta = self._load_metadata(session_id)
        return meta.model_keys if meta else []

    # --- Panel Storage ---

    def store_panel(
        self,
        session_id: str,
        key: str,
        panel: Any,
    ) -> None:
        """Store a StatisticsPanel object."""
        path = self.base_dir / session_id / "panels" / f"{key}.pkl"
        self._atomic_pickle_dump(panel, path)

        meta = self._load_metadata(session_id)
        if meta and key not in meta.panel_keys:
            meta.panel_keys.append(key)
            self._save_metadata(session_id, meta)
        self._touch_session(session_id)

    def load_panel(self, session_id: str, key: str) -> Optional[Any]:
        """Load a StatisticsPanel object."""
        path = self.base_dir / session_id / "panels" / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self._touch_session(session_id)
                return pickle.load(f)
        return None

    def list_panels(self, session_id: str) -> List[str]:
        """List stored panel keys for a session."""
        meta = self._load_metadata(session_id)
        return meta.panel_keys if meta else []

    # --- Cache Results ---

    def cache_result(
        self,
        session_id: str,
        key: str,
        result: Any,
    ) -> None:
        """
        Cache a computation result in session metadata.

        Parameters
        ----------
        session_id : str
            Session to cache in.
        key : str
            Cache key.
        result : Any
            Result to cache (must be JSON-serializable).
        """
        meta = self._load_metadata(session_id)
        if meta:
            meta.results_cache[key] = result
            self._save_metadata(session_id, meta)

    def get_cached_result(
        self,
        session_id: str,
        key: str,
    ) -> Optional[Any]:
        """
        Get cached result.

        Parameters
        ----------
        session_id : str
            Session to get from.
        key : str
            Cache key.

        Returns
        -------
        Any or None
            Cached result, or None if not found.
        """
        meta = self._load_metadata(session_id)
        if meta:
            return meta.results_cache.get(key)
        return None

    def clear_cache(self, session_id: str) -> None:
        """Clear all cached results for a session."""
        meta = self._load_metadata(session_id)
        if meta:
            meta.results_cache = {}
            self._save_metadata(session_id, meta)

    # --- Internal Methods ---

    def _save_metadata(self, session_id: str, meta: SessionMetadata) -> None:
        """Save session metadata to disk."""
        path = self.base_dir / session_id / "metadata.json"
        with open(path, "w") as f:
            json.dump(meta.__dict__, f, indent=2)
        self._active_sessions[session_id] = meta

    def _load_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata from disk."""
        # Check cache first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        path = self.base_dir / session_id / "metadata.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            # Handle missing fields for backwards compatibility
            data.setdefault("panel_keys", [])
            data.setdefault("results_cache", {})
            meta = SessionMetadata(**data)
            self._active_sessions[session_id] = meta
            return meta
        return None

    def _touch_session(self, session_id: str) -> None:
        """Update session last_accessed timestamp."""
        meta = self._load_metadata(session_id)
        if meta:
            meta.last_accessed = datetime.now().isoformat()
            self._save_metadata(session_id, meta)

    @staticmethod
    def _atomic_pickle_dump(obj: Any, path: Path) -> None:
        """Pickle ``obj`` to ``path`` atomically.

        Writes to a sibling temp file first and then uses ``os.replace`` to
        atomically swap it into place. This prevents truncated / partial
        pickle files when two tool calls update the same data or panel
        concurrently -- the symptom was ``pickle.UnpicklingError: Ran out
        of input`` on a subsequent read, because one writer had truncated
        the file after another had started writing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per-call temp name: pid + thread id + uuid. Needed because
        # two threads in the same process share the same pid, so a
        # pid-only suffix would collide under in-process concurrency.
        tmp_suffix = (
            f".tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
        )
        tmp_path = path.with_suffix(path.suffix + tmp_suffix)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync is best-effort -- some filesystems reject it
                    pass
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp file so we don't leak partial writes
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session."""
        meta = self._load_metadata(session_id)
        if meta:
            return {
                "session_id": meta.session_id,
                "created_at": meta.created_at,
                "last_accessed": meta.last_accessed,
                "data_keys": meta.data_keys,
                "graph_keys": meta.graph_keys,
                "model_keys": meta.model_keys,
                "panel_keys": meta.panel_keys,
                "n_cached_results": len(meta.results_cache),
            }
        return None
