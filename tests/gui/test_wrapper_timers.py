import time
import tkinter as tk
from unittest.mock import MagicMock

import pytest

from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin
from countess.gui.main import ConfiguratorWrapper, MainWindow, make_root
from countess.gui.tabular import TabularDataFrame


class PreviewPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1
    show_preview = True

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        return ddbc.sql("select range as x from range(10)")


def descendants(widget):
    for w in widget.winfo_children():
        yield w
        yield from descendants(w)


def pump(root, seconds):
    """Run the Tk event loop for a while, as mainloop() would."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.01)


def wait_for(root, condition, seconds=10):
    deadline = time.time() + seconds
    while not condition():
        assert time.time() < deadline, "timed out"
        root.update()
        time.sleep(0.01)


@pytest.fixture(name="root")
def fixture_root():
    root = make_root()
    root.callback_errors = []
    root.report_callback_exception = lambda *exc: root.callback_errors.append(exc)
    yield root
    root.destroy()


@pytest.fixture(name="graph_node")
def fixture_graph_node():
    graph = PipelineGraph()
    node = PipelineNode(name="n", plugin=PreviewPlugin())
    graph.add_node(node)
    yield graph, node
    graph.stop_thread(timeout=10)


def settle(root, graph):
    graph.reset()
    wait_for(root, lambda: not graph._worker_running)  # pylint: disable=protected-access


@pytest.mark.gui
def test_config_changes_are_debounced(root, graph_node):
    graph, node = graph_node
    callback = MagicMock()
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, callback)

    for _ in range(20):
        wrap.config_change_callback()
        root.update()

    callback.assert_not_called()  # nothing yet: still waiting for the changes to stop
    wait_for(root, lambda: callback.call_count > 0)
    pump(root, 0.5)

    callback.assert_called_once_with(node, True)
    wrap.destroy()
    assert not root.callback_errors


@pytest.mark.gui
def test_only_one_poll_chain(root, graph_node):
    graph, node = graph_node
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    node.status = PipelineNodeStatus.WORKING  # never finishes, so polling continues

    for _ in range(10):
        wrap.notify()
        wrap.config_change_callback()
        pump(root, 0.3)

    # a poll, and (while something is changing) a debounce: never a pile of them
    assert len(wrap._after_ids) <= 3  # pylint: disable=protected-access
    wrap.destroy()


@pytest.mark.gui
def test_destroy_cancels_pending_callbacks(root, graph_node):
    graph, node = graph_node
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    wrap.config_change_callback()
    wrap.notify()
    assert wrap._after_ids  # pylint: disable=protected-access

    wrap.destroy()
    assert not wrap._after_ids  # pylint: disable=protected-access

    pump(root, 0.8)  # nothing should fire against the destroyed widgets
    assert not root.callback_errors


@pytest.mark.gui
def test_preview_is_not_duplicated(root, graph_node):
    graph, node = graph_node
    settle(root, graph)
    assert node.status == PipelineNodeStatus.RESULT

    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    for _ in range(5):
        wrap.replace_preview_subframe()
        wrap.show_preview_subframe()
    pump(root, 1.0)

    previews = [w for w in descendants(wrap.frame) if isinstance(w, TabularDataFrame)]
    assert len(previews) == 1
    wrap.destroy()
    assert not root.callback_errors


@pytest.mark.gui
def test_preview_copes_with_dropped_table(root, graph_node):
    """The worker may drop a table between the GUI seeing RESULT and reading it."""
    graph, node = graph_node
    settle(root, graph)
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    pump(root, 0.2)
    wrap.hide_preview_subframe()

    graph.ddbc.sql(f"DROP TABLE {node.table_name}")
    wrap.show_preview_subframe()  # status still says RESULT, but the table has gone

    assert wrap.preview_subframe is None
    assert wrap._preview_after_id is not None  # pylint: disable=protected-access

    # once it has been recalculated the preview appears
    settle(root, graph)
    wait_for(root, lambda: wrap.preview_subframe is not None)
    wrap.destroy()
    assert not root.callback_errors


@pytest.mark.gui
def test_new_and_load_discard_old_graph(root):
    mw = MainWindow(root, "tests/simple.ini")
    pump(root, 0.2)

    old_graph = mw.graph
    node = next(iter(old_graph.traverse_nodes()))
    mw.node_select(node)
    assert mw.config_wrapper_cache
    wrapper = mw.config_wrapper

    mw.config_new()
    assert mw.graph is not old_graph
    assert not mw.config_wrapper_cache
    assert mw.config_wrapper is not wrapper
    assert old_graph._quit  # pylint: disable=protected-access
    assert wrapper._destroyed  # pylint: disable=protected-access
    pump(root, 0.8)

    mw.config_load("tests/simple.ini")
    pump(root, 0.5)
    assert not mw.config_wrapper_cache
    assert not root.callback_errors

    mw.graph.stop_thread(timeout=10)


@pytest.mark.gui
def test_no_nested_event_loop_in_poll_done(root, graph_node):
    """config_change_poll_done must not run other event handlers (as Tk's update() does)."""
    graph, node = graph_node
    settle(root, graph)
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    pump(root, 0.8)

    ran = []
    root.after(0, lambda: ran.append(True))
    time.sleep(0.05)  # make the timer due
    wrap.config_change_poll_done()
    assert not ran  # it only runs when we return to the event loop
    wrap.destroy()
    assert isinstance(root, tk.Tk)
