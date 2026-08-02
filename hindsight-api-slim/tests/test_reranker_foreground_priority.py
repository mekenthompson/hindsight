"""
Deterministic tests for the local reranker's foreground-priority lane.

The local cross-encoder runs CPU-bound reranks in a shared worker pool. A plain
FIFO ``ThreadPoolExecutor`` lets an interactive (foreground) rerank queue behind
a large fan-out of background reranks (issued by consolidation/reflect
sub-recalls), spiking foreground recall latency. ``_PriorityRerankExecutor``
fixes this by dispatching foreground work ahead of *queued* background work.

These tests assert **ordering / reservation semantics**, not wall-clock timing:
there are no sleeps. ``threading.Event`` barriers and short ``result()`` timeouts
are used only as correctness bounds (correct code completes instantly; only a
regression to a plain FIFO pool would hit them).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import AsyncMock

from hindsight_api.engine.cross_encoder import (
    LocalSTCrossEncoder,
    _PriorityRerankExecutor,
)
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult


# ---------------------------------------------------------------------------
# _PriorityRerankExecutor — the core ordering guarantee
# ---------------------------------------------------------------------------


def test_foreground_dispatched_before_queued_background():
    """A foreground job submitted AFTER background jobs is dispatched first.

    With a single worker held busy, every subsequent submission queues. The
    priority queue must drain the foreground job ahead of the earlier-queued
    background jobs. A plain FIFO pool would run them in submission order
    (bg1, bg2, fg, bg3) and fail this assertion.
    """
    ex = _PriorityRerankExecutor(max_workers=1)
    try:
        gate = threading.Event()
        worker_busy = threading.Event()
        order: list[str] = []
        order_lock = threading.Lock()

        def occupy():
            worker_busy.set()
            # Hold the only worker until we have queued everything behind it.
            # This is a synchronization barrier, not a timing sleep.
            gate.wait(timeout=5)
            return "occupy"

        def record(tag: str):
            with order_lock:
                order.append(tag)
            return tag

        f_occupy = ex.submit(occupy)
        assert worker_busy.wait(timeout=5), "worker never picked up the occupying job"

        # Queue background jobs, then a foreground job, then more background.
        f_bg1 = ex.submit(record, "bg1", background=True)
        f_bg2 = ex.submit(record, "bg2", background=True)
        f_fg = ex.submit(record, "fg", background=False)
        f_bg3 = ex.submit(record, "bg3", background=True)

        # Release the worker; the queue now drains in priority order.
        gate.set()
        for f in (f_occupy, f_bg1, f_bg2, f_fg, f_bg3):
            f.result(timeout=5)

        # Foreground jumped the queue; background preserved FIFO among itself.
        assert order == ["fg", "bg1", "bg2", "bg3"], order
    finally:
        ex.shutdown()


def test_same_priority_preserves_fifo():
    """Within a priority class, submission order (FIFO) is preserved."""
    ex = _PriorityRerankExecutor(max_workers=1)
    try:
        gate = threading.Event()
        worker_busy = threading.Event()
        order: list[str] = []
        order_lock = threading.Lock()

        def occupy():
            worker_busy.set()
            gate.wait(timeout=5)

        def record(tag: str):
            with order_lock:
                order.append(tag)

        f_occupy = ex.submit(occupy)
        assert worker_busy.wait(timeout=5)

        futures = [f_occupy]
        for tag in ("fg1", "fg2", "fg3"):
            futures.append(ex.submit(record, tag, background=False))

        gate.set()
        for f in futures:
            f.result(timeout=5)

        assert order == ["fg1", "fg2", "fg3"], order
    finally:
        ex.shutdown()


def test_submitted_callable_exception_propagates_to_future():
    """A raising job surfaces its exception via the future, not the worker thread."""
    ex = _PriorityRerankExecutor(max_workers=1)
    try:

        def boom():
            raise ValueError("kaboom")

        fut = ex.submit(boom)
        try:
            fut.result(timeout=5)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert str(e) == "kaboom"
    finally:
        ex.shutdown()


# ---------------------------------------------------------------------------
# LocalSTCrossEncoder.predict — plumbing the background flag to the pool
# ---------------------------------------------------------------------------


class _SpyExecutor:
    """Records the ``background`` flag each submit was called with."""

    def __init__(self):
        self.calls: list[bool] = []

    def submit(self, fn, /, *args, background: bool = False) -> Future:
        self.calls.append(background)
        fut: Future = Future()
        # Return a shaped result without running the (mocked) model.
        fut.set_result([0.5] * len(args[0]))
        return fut


async def test_predict_forwards_background_flag_to_executor():
    """predict(background=...) must reach the priority executor unchanged."""
    encoder = LocalSTCrossEncoder(model_name="test-model")
    encoder._model = object()  # non-None so predict() does not raise

    spy = _SpyExecutor()
    saved = LocalSTCrossEncoder._executor
    LocalSTCrossEncoder._executor = spy  # _get_executor() returns this when set
    try:
        await encoder.predict([("q", "d")], background=True)
        await encoder.predict([("q", "d")], background=False)
        await encoder.predict([("q", "d")])  # default is foreground
        assert spy.calls == [True, False, False]
    finally:
        LocalSTCrossEncoder._executor = saved


# ---------------------------------------------------------------------------
# CrossEncoderReranker.rerank — plumbing background to predict
# ---------------------------------------------------------------------------


def _make_candidates(n: int) -> list[MergedCandidate]:
    candidates = []
    for i in range(n):
        retrieval = RetrievalResult(
            id=f"id-{i}",
            text=f"Document {i}",
            fact_type="world",
            occurred_start=None,
            occurred_end=None,
        )
        candidates.append(MergedCandidate(retrieval=retrieval, rrf_score=1.0 / (i + 1)))
    return candidates


async def test_reranker_forwards_background_to_predict():
    """CrossEncoderReranker.rerank forwards its background flag to predict."""
    ce = AsyncMock()
    ce.predict = AsyncMock(return_value=[0.1, 0.2])
    ce.provider_name = "local"
    ce.initialize = AsyncMock()

    reranker = CrossEncoderReranker(cross_encoder=ce)
    reranker._initialized = True

    candidates = _make_candidates(2)
    await reranker.rerank("q", candidates, background=True)
    assert ce.predict.await_args.kwargs.get("background") is True

    await reranker.rerank("q", candidates)  # default
    assert ce.predict.await_args.kwargs.get("background") is False
