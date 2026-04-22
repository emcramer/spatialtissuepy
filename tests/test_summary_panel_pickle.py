"""
Regression tests for pickling StatisticsPanel / MetricInfo.

Background
----------
Panels used to fail to pickle with

    PicklingError: Can't pickle <function cell_counts at 0x...>: it's not
    the same object as spatialtissuepy.summary.population.cell_counts

because ``register()`` stores the raw ``func`` in ``MetricInfo`` while
returning a ``functools.wraps`` wrapper as the module attribute, so
pickle's qualname-based function lookup found a different object than the
one being serialised. Parallel writes to the same panel file also
produced truncated pickle blobs, surfacing on the next read as
``Ran out of input``.

The fix adds ``MetricInfo.__reduce__`` so panels pickle metric entries by
*name* and re-resolve from the live registry on unpickle, plus an atomic
``_atomic_pickle_dump`` helper in ``SessionManager`` that writes through
a temp file + ``os.replace`` so parallel writers cannot leave a partial
file on disk.
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Dict

import pytest

from spatialtissuepy.summary import (
    StatisticsPanel,
    load_panel,
    register_custom_metric,
    unregister_custom_metric,
)
from spatialtissuepy.summary.registry import get_metric


class TestPanelPickle:
    """Panels should pickle cleanly regardless of how metrics were registered."""

    def test_builtin_panel_roundtrips(self):
        panel = load_panel("comprehensive")
        assert panel.n_metrics > 0

        blob = pickle.dumps(panel)
        restored = pickle.loads(blob)

        assert restored.n_metrics == panel.n_metrics
        assert [m.name for m in restored.metrics] == [m.name for m in panel.metrics]

    @pytest.mark.parametrize("preset", ["basic", "spatial", "neighborhood", "comprehensive"])
    def test_all_preset_panels_pickle(self, preset: str):
        panel = load_panel(preset)
        blob = pickle.dumps(panel)
        restored = pickle.loads(blob)
        assert restored.n_metrics == panel.n_metrics

    def test_registered_custom_metric_roundtrips(self):
        name = "test_pickle_ratio_unique"

        @register_custom_metric(name=name, description="Test custom metric")
        def _custom(data) -> Dict[str, float]:
            return {name: 0.5}

        try:
            panel = StatisticsPanel(name="test")
            panel.add(name)
            blob = pickle.dumps(panel)
            restored = pickle.loads(blob)
            assert restored.n_metrics == 1
            assert restored.metrics[0].name == name
        finally:
            unregister_custom_metric(name)

    def test_inline_metric_refuses_to_pickle(self):
        """Inline functions live only in the panel and can't survive a pickle round-trip."""
        panel = StatisticsPanel(name="inline")

        def _inline(data) -> Dict[str, float]:
            return {"v": 1.0}

        panel.add_custom_function("inline_unpicklable", _inline)

        with pytest.raises(TypeError, match="not in the global registry"):
            pickle.dumps(panel)

    def test_missing_registry_entry_fails_loudly_on_unpickle(self):
        """Unpickling a panel whose custom metric isn't re-registered raises a clear error."""
        name = "test_unpickle_missing_unique"

        @register_custom_metric(name=name, description="Test")
        def _m(data) -> Dict[str, float]:
            return {name: 1.0}

        panel = StatisticsPanel(name="test")
        panel.add(name)
        blob = pickle.dumps(panel)

        unregister_custom_metric(name)
        with pytest.raises(RuntimeError, match="no metric with that name is registered"):
            pickle.loads(blob)


class TestAtomicPickleDump:
    """Concurrent writes must never leave a truncated pickle on disk."""

    def test_atomic_write_survives_concurrent_writers(self, tmp_path: Path):
        from spatialtissuepy.mcp.session import SessionManager

        path = tmp_path / "payload.pkl"
        payloads = [
            {"writer": i, "data": list(range(1000))} for i in range(8)
        ]

        def write(p):
            SessionManager._atomic_pickle_dump(p, path)

        threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Whichever writer won, the result must be a valid, fully-written pickle
        loaded = pickle.loads(path.read_bytes())
        assert loaded["writer"] in {p["writer"] for p in payloads}
        assert loaded["data"] == list(range(1000))

    def test_failed_write_leaves_no_partial_file(self, tmp_path: Path):
        from spatialtissuepy.mcp.session import SessionManager

        path = tmp_path / "will_fail.pkl"

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("intentional failure")

        with pytest.raises(RuntimeError, match="intentional"):
            SessionManager._atomic_pickle_dump(Unpicklable(), path)

        assert not path.exists(), "failed write should not have produced a file"
        # Temp file should also be cleaned up
        siblings = list(tmp_path.iterdir())
        assert all("tmp" not in s.name for s in siblings), f"temp file leaked: {siblings}"


class TestPanelSessionRoundtrip:
    """End-to-end: SessionManager can store and retrieve a panel with metrics."""

    def test_store_and_load_builtin_panel(self, tmp_path: Path):
        from spatialtissuepy.mcp.session import SessionManager

        mgr = SessionManager(base_dir=tmp_path)
        sid = mgr.create_session()

        panel = load_panel("basic")
        mgr.store_panel(sid, "fingerprint", panel)

        restored = mgr.load_panel(sid, "fingerprint")
        assert restored is not None
        assert restored.n_metrics == panel.n_metrics
        assert [m.name for m in restored.metrics] == [m.name for m in panel.metrics]
