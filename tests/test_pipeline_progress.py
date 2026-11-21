import threading
import time
from typing import Optional
from unittest.mock import MagicMock

import duckdb
import pytest

from countess.core.parameters import IntegerParam
from countess.core.pipeline import GraphProgress, NodeProgress, PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin

TIMEOUT = 15


class GatedPlugin(DuckdbPlugin):
    """Runs until it's let go, for looking at what happens meanwhile."""

    version = "0"
    num_inputs = 0
    num_outputs = 1
    param = IntegerParam("param", 1)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.started = threading.Event()
        self.release = threading.Event()

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        self.started.set()
        assert self.release.wait(TIMEOUT)
        return ddbc.sql("select 1 as x")


class SortPlugin(DuckdbPlugin):
    """Takes a while, in DuckDB, so that there's progress to report."""

    version = "0"
    num_inputs = 0
    num_outputs = 1

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        return ddbc.sql("select * from src order by s, b")


def wait_idle(graph):
    deadline = time.time() + TIMEOUT
    while graph._worker_running:  # pylint: disable=protected-access
        assert time.time() < deadline, "worker did not finish"
        time.sleep(0.01)


@pytest.fixture(name="graph")
def fixture_graph():
    graph = PipelineGraph()
    yield graph
    graph.stop_thread(timeout=TIMEOUT)


def test_nothing_going_on(graph):
    graph.add_node(PipelineNode("n", plugin=GatedPlugin()))
    progress = graph.get_progress()
    assert progress == GraphProgress([], 0)
    assert not progress.active


def test_waiting_counts_only_nodes_which_will_run(graph):
    blocker = PipelineNode("a blocker", plugin=GatedPlugin())
    graph.add_node(blocker)
    queued = [PipelineNode(f"queued {i}", plugin=GatedPlugin()) for i in range(3)]
    for node in queued:
        node.plugin.release.set()
        graph.add_node(node)
    graph.add_node(PipelineNode("NEW"))

    # make the blocker the one which runs first, whatever the order they're sorted in
    for node in queued:
        node.status = PipelineNodeStatus.RESULT
    blocker.plugin.release.clear()
    graph.update()
    assert blocker.plugin.started.wait(TIMEOUT)
    for node in queued:
        node.mark_dirty()

    progress = graph.get_progress()
    assert [r.node for r in progress.running] == [blocker]
    assert progress.waiting == 3  # not 4: the node with no plugin isn't waiting for anything

    assert progress.running[0].percent is None  # in python, not in a query: nothing to report

    blocker.plugin.release.set()
    wait_idle(graph)
    assert graph.get_progress() == GraphProgress([], 0)


@pytest.mark.parametrize(
    "reported,expected",
    [(42, 42.0), (0, 0.0), (0.5, 0.5), (100, 100.0), (-1, None), (-1.0, None), (None, None), (250, None)],
)
def test_percent_from_query_progress(graph, reported, expected):
    node = PipelineNode("n", plugin=GatedPlugin())
    node.plugin.query_progress = MagicMock(return_value=reported)
    graph.add_node(node)
    node.status = PipelineNodeStatus.WORKING
    graph._worker_running = True  # pylint: disable=protected-access
    graph._current_node = node  # pylint: disable=protected-access
    try:
        progress = graph.get_progress()
    finally:
        graph._worker_running = False  # pylint: disable=protected-access
        graph._current_node = None  # pylint: disable=protected-access
    assert progress.running == [NodeProgress(node, expected)]
    node.plugin.query_progress.assert_called_once_with(graph.thread_cursor)


def test_a_query_which_cant_be_asked_is_just_unknown(graph):
    node = PipelineNode("n", plugin=GatedPlugin())
    node.plugin.query_progress = MagicMock(side_effect=duckdb.ConnectionException("closed"))
    graph.add_node(node)
    node.status = PipelineNodeStatus.WORKING
    graph._worker_running = True  # pylint: disable=protected-access
    graph._current_node = node  # pylint: disable=protected-access
    try:
        assert graph.get_progress().running == [NodeProgress(node, None)]
    finally:
        graph._worker_running = False  # pylint: disable=protected-access
        graph._current_node = None  # pylint: disable=protected-access


def test_only_the_current_node_is_asked(graph):
    """The worker's cursor is running one query, so it's no use asking about any other node."""
    node = PipelineNode("n", plugin=GatedPlugin())
    node.plugin.query_progress = MagicMock(return_value=50)
    graph.add_node(node)
    node.status = PipelineNodeStatus.WORKING  # ... but it isn't the node the worker is on
    graph._worker_running = True  # pylint: disable=protected-access
    try:
        assert graph.get_progress().running == [NodeProgress(node, None)]
    finally:
        graph._worker_running = False  # pylint: disable=protected-access
    node.plugin.query_progress.assert_not_called()


def test_get_progress_while_the_graph_is_edited_and_run(graph):
    errors = []
    stop = threading.Event()

    def editor():
        try:
            while not stop.is_set():
                node = PipelineNode("x", plugin=GatedPlugin())
                node.plugin.release.set()
                graph.add_node(node)
                graph.update()
                time.sleep(0.001)
                graph.del_node(node)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    thread = threading.Thread(target=editor)
    thread.start()
    try:
        deadline = time.time() + 1.5
        while time.time() < deadline:
            graph.get_progress()
    finally:
        stop.set()
        thread.join(TIMEOUT)
    assert not errors


def test_real_query_reports_progress(graph):
    """Against DuckDB itself: while a query runs its progress is a percentage."""
    graph.ddbc.sql(
        "create table src as select range as a, (random()*1e9)::bigint as b, md5(range::varchar) as s from range(3000000)"
    )
    node = PipelineNode("sort", plugin=SortPlugin())
    graph.add_node(node)
    graph.update()

    seen: list[Optional[float]] = []
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        progress = graph.get_progress()
        seen.extend(r.percent for r in progress.running)
        if not progress.active and node.status == PipelineNodeStatus.RESULT:
            break
        time.sleep(0.01)

    assert node.status == PipelineNodeStatus.RESULT
    assert seen, "never saw the node running"
    figures = [p for p in seen if p is not None]
    assert figures, "DuckDB never said how far the query had got"
    assert all(0 <= p <= 100 for p in figures)
    assert graph.get_progress() == GraphProgress([], 0)
