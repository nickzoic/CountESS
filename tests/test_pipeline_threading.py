"""Tests for the interaction between the GUI thread (which edits nodes, links
and parameters) and the PipelineGraph worker thread (which recalculates them)."""

import threading
import time
from unittest.mock import MagicMock

import duckdb
import pytest

from countess.core.parameters import IntegerParam
from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin

TIMEOUT = 10


class GatedPlugin(DuckdbPlugin):
    """An input plugin whose result is `param`, and which can be held up
    part way through running in order to make changes while it is running."""

    version = "0"
    num_inputs = 0
    num_outputs = 1
    param = IntegerParam("param", 1)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.runs = 0
        self.interrupt_next = False

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        self.runs += 1
        value = self.param.value
        self.started.set()
        assert self.release.wait(TIMEOUT)
        if self.interrupt_next:
            self.interrupt_next = False
            raise duckdb.InterruptException("interrupted")
        return ddbc.sql(f"select {int(value)} as x")


def wait_idle(graph: PipelineGraph):
    deadline = time.time() + TIMEOUT
    while graph._worker_running:  # pylint: disable=protected-access
        assert time.time() < deadline, "worker did not finish"
        time.sleep(0.01)


def table_value(graph: PipelineGraph, node: PipelineNode):
    return graph.ddbc.sql(f"select x from {node.table_name}").fetchall()[0][0]


@pytest.fixture(name="graph")
def fixture_graph():
    graph = PipelineGraph()
    yield graph
    graph.stop_thread(timeout=TIMEOUT)


def test_edit_during_run_is_not_overwritten():
    """mark_dirty() while run() is in progress must win over the result of that run."""
    graph = PipelineGraph()
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    node.plugin.release.clear()

    result = []
    thread = threading.Thread(target=lambda: result.append(node.run(graph.thread_cursor)))
    thread.start()
    assert node.plugin.started.wait(TIMEOUT)
    assert node.status == PipelineNodeStatus.WORKING

    node.mark_dirty()
    node.plugin.release.set()
    thread.join(TIMEOUT)

    assert node.status == PipelineNodeStatus.DIRTY
    assert result == [True]  # "needs another go"
    graph.stop_thread()


def test_error_during_edit_is_not_reported():
    """If the node is changed while running, an exception from the run (which was
    probably caused by the half-edited state) shouldn't be shown as an error."""
    graph = PipelineGraph()
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    node.plugin.release.clear()

    thread = threading.Thread(target=lambda: node.run(None))  # ddbc=None: fails
    thread.start()
    assert node.plugin.started.wait(TIMEOUT)
    node.mark_dirty()
    node.plugin.release.set()
    thread.join(TIMEOUT)

    assert node.status == PipelineNodeStatus.DIRTY
    assert not node.message
    graph.stop_thread()


def test_interrupted_node_is_left_dirty():
    graph = PipelineGraph()
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)

    node.plugin.interrupt_next = True
    assert node.run(graph.thread_cursor) is True
    assert node.status == PipelineNodeStatus.DIRTY

    assert node.run(graph.thread_cursor) is False
    assert node.status == PipelineNodeStatus.RESULT
    graph.stop_thread()


def test_update_does_not_block(graph):
    """update() is called from the GUI thread so mustn't wait for a running query."""
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    node.plugin.release.clear()

    graph.update()
    assert node.plugin.started.wait(TIMEOUT)

    start = time.time()
    for _ in range(20):
        graph.update()
    assert time.time() - start < 1.0

    node.plugin.release.set()
    wait_idle(graph)
    assert node.status == PipelineNodeStatus.RESULT


def test_change_during_run_gets_recalculated(graph):
    """End to end: edit a parameter while the worker is mid-calculation, and the
    final result must reflect the edit (not the value from before it)."""
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    node.plugin.release.clear()

    graph.update()
    assert node.plugin.started.wait(TIMEOUT)

    node.plugin.param.value = 2
    node.mark_dirty()
    graph.update()
    node.plugin.release.set()
    wait_idle(graph)

    assert node.status == PipelineNodeStatus.RESULT
    assert table_value(graph, node) == 2
    assert node.plugin.runs >= 2


def test_children_follow_parents(graph):
    parent = PipelineNode("parent", plugin=GatedPlugin())
    child = PipelineNode("child", plugin=GatedPlugin())
    graph.add_node(parent)
    graph.add_node(child)

    graph.reset()
    wait_idle(graph)
    assert parent.status == child.status == PipelineNodeStatus.RESULT

    parent.plugin.param.value = 5
    parent.mark_dirty()
    graph.update()
    wait_idle(graph)
    assert table_value(graph, parent) == 5


def test_worker_exits_when_idle(graph):
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    graph.update()
    wait_idle(graph)
    assert graph.thread is not None
    graph.thread.join(TIMEOUT)
    assert not graph.thread.is_alive()

    # ... and update() starts a new one.
    node.plugin.param.value = 3
    node.mark_dirty()
    graph.update()
    wait_idle(graph)
    assert table_value(graph, node) == 3


def test_stop_thread_interrupts_and_returns(graph):
    node = PipelineNode("n", plugin=GatedPlugin())
    graph.add_node(node)
    node.plugin.release.clear()
    graph.update()
    assert node.plugin.started.wait(TIMEOUT)

    # stop_thread(wait=False) must not block even though the worker is busy
    start = time.time()
    graph.stop_thread(wait=False)
    assert time.time() - start < 1.0

    node.plugin.release.set()
    wait_idle(graph)


def test_traverse_nodes_with_cycle_terminates():
    graph = PipelineGraph()
    a = PipelineNode("a", plugin=GatedPlugin())
    b = PipelineNode("b", plugin=GatedPlugin())
    c = PipelineNode("c", plugin=GatedPlugin())
    for n in (a, b, c):
        n.plugin.num_inputs = 1
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    b.add_parent(a)
    c.add_parent(b)
    a.add_parent(c)  # cycle

    assert list(graph.traverse_nodes()) == []
    assert list(graph.traverse_nodes_backwards()) == []
    a.mark_dirty()  # mark_dirty must not recurse forever either
    graph.stop_thread()


def test_graph_edited_while_traversed():
    """Edit the graph from one thread while another walks it: no exceptions."""
    graph = PipelineGraph()
    base = PipelineNode("base")
    graph.add_node(base)
    errors = []
    stop = threading.Event()

    def editor():
        try:
            while not stop.is_set():
                node = PipelineNode("x")
                graph.add_node(node)
                node.add_parent(base)
                node.del_parent(base)
                graph.del_node(node)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    thread = threading.Thread(target=editor)
    thread.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            list(graph.traverse_nodes())
            list(graph.traverse_nodes_backwards())
    finally:
        stop.set()
        thread.join(TIMEOUT)

    assert not errors
    graph.stop_thread()


def test_load_config_applies_once():
    plugin = GatedPlugin()
    plugin.set_parameter = MagicMock()
    node = PipelineNode("n", plugin=plugin)
    node.set_config("param", "1", ".")

    threads = [threading.Thread(target=node.load_config) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)

    plugin.set_parameter.assert_called_once()


def test_empty_result_is_still_an_error():
    """(existing behaviour, preserved) a plugin which produces no rows is flagged."""
    graph = PipelineGraph()

    class Empty(GatedPlugin):
        def execute_multi(self, ddbc, sources, row_limit=None):
            return ddbc.sql("select 1 as x where false")

    node = PipelineNode("n", plugin=Empty())
    graph.add_node(node)
    node.run(graph.thread_cursor)
    assert node.status == PipelineNodeStatus.ERROR
    graph.stop_thread()
