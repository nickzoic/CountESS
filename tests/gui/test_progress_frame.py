import threading
import time
import tkinter as tk
from typing import Optional

import pytest

from countess.core.pipeline import GraphProgress, NodeProgress, PipelineNode
from countess.core.plugins import DuckdbPlugin
from countess.gui.main import MainWindow, QueryProgressFrame, make_root


def pump(root, seconds=0.1):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.005)


def wait_for(root, condition, seconds=15):
    deadline = time.time() + seconds
    while not condition():
        assert time.time() < deadline, "timed out"
        root.update()
        time.sleep(0.005)


class FakeGraph:
    def __init__(self):
        self.progress = GraphProgress([], 0)

    def get_progress(self):
        return self.progress


def node_progress(name: str, percent: Optional[float]) -> NodeProgress:
    return NodeProgress(PipelineNode(name), percent)


def bar_text(bar) -> str:
    return bar.style.lookup(bar.style_name, "text")


@pytest.fixture(name="root")
def fixture_root():
    root = make_root()
    root.geometry("800x400")
    root.callback_errors = []
    root.report_callback_exception = lambda *exc: root.callback_errors.append(exc)
    yield root
    root.destroy()


@pytest.fixture(name="setup")
def fixture_setup(root):
    graph = FakeGraph()
    frame = QueryProgressFrame(root, lambda: graph)
    frame.grid(row=0, column=0, sticky=tk.EW)
    frame.SHOW_AFTER = 0
    root.update()
    return root, graph, frame


def visible_bars(frame):
    return [w for w in frame.pack_slaves() if w is not frame.waiting_label]


@pytest.mark.gui
def test_nothing_shown_when_nothing_is_happening(setup):
    root, _, frame = setup
    frame.show_progress(GraphProgress([], 0))
    frame.show_progress(None)
    root.update()
    assert not frame.shown and not frame.winfo_ismapped()


@pytest.mark.gui
def test_determinate_bar_with_percentage(setup):
    root, _, frame = setup
    frame.show_progress(GraphProgress([node_progress("Load", 42.4)], 0))
    root.update()
    assert frame.shown and frame.winfo_ismapped()
    (bar,) = visible_bars(frame)
    assert bar_text(bar) == "Load: 42%"
    assert str(bar["mode"]) == "determinate" and float(bar["value"]) == pytest.approx(42.4)
    assert frame.waiting_label not in frame.pack_slaves()


@pytest.mark.gui
def test_indeterminate_bar_when_progress_unknown(setup):
    root, _, frame = setup
    frame.show_progress(GraphProgress([node_progress("Load", None)], 0))
    root.update()
    (bar,) = visible_bars(frame)
    assert str(bar["mode"]) == "indeterminate"
    assert bar_text(bar) == "Load: working…"

    frame.show_progress(GraphProgress([node_progress("Load", 10)], 0))  # ... and back again
    assert str(bar["mode"]) == "determinate" and bar_text(bar) == "Load: 10%"


@pytest.mark.gui
def test_waiting_nodes(setup):
    root, _, frame = setup
    frame.show_progress(GraphProgress([node_progress("A", 5)], 3))
    root.update()
    assert frame.waiting_label["text"] == "3 more waiting"
    assert frame.waiting_label in frame.pack_slaves()

    frame.show_progress(GraphProgress([], 2))  # between one node and the next
    assert frame.shown and visible_bars(frame) == []
    assert frame.waiting_label["text"] == "2 waiting"

    frame.show_progress(GraphProgress([node_progress("A", 5)], 0))
    assert frame.waiting_label not in frame.pack_slaves()


@pytest.mark.gui
def test_goes_away_when_finished(setup):
    root, _, frame = setup
    frame.show_progress(GraphProgress([node_progress("A", 50)], 1))
    root.update()
    assert frame.winfo_ismapped()
    frame.show_progress(GraphProgress([], 0))
    root.update()
    assert not frame.shown and not frame.winfo_ismapped()

    frame.show_progress(GraphProgress([node_progress("B", 1)], 0))  # and comes back where it was
    root.update()
    assert frame.winfo_ismapped() and frame.grid_info()["row"] == 0


@pytest.mark.gui
def test_quick_queries_dont_flash_up(setup):
    root, _, frame = setup
    frame.SHOW_AFTER = 0.3
    frame.show_progress(GraphProgress([node_progress("A", 0)], 0))
    root.update()
    assert not frame.shown  # not yet

    frame.show_progress(GraphProgress([], 0))  # it finished in time: nothing was ever shown
    time.sleep(0.35)
    frame.show_progress(GraphProgress([node_progress("B", 0)], 0))
    assert not frame.shown  # and the clock started again

    time.sleep(0.35)
    frame.show_progress(GraphProgress([node_progress("B", 20)], 0))
    assert frame.shown  # but something slow does appear


@pytest.mark.gui
def test_bars_are_reused(setup):
    root, _, frame = setup
    for i in range(20):
        frame.show_progress(GraphProgress([node_progress(f"node {i}", i)], 0))
    assert len(frame.bars) == 1  # (each one would make Tk keep a style for ever)

    frame.show_progress(GraphProgress([node_progress("a", 1), node_progress("b", 2)], 0))
    assert len(frame.bars) == 2 and len(visible_bars(frame)) == 2
    frame.show_progress(GraphProgress([node_progress("a", 3)], 0))
    assert len(frame.bars) == 2 and len(visible_bars(frame)) == 1


@pytest.mark.gui
def test_polls_the_graph(setup):
    root, graph, frame = setup
    frame.POLL_MS = 20
    graph.progress = GraphProgress([node_progress("Polled", 33)], 0)
    wait_for(root, lambda: frame.shown)
    assert bar_text(frame.bars[0]) == "Polled: 33%"

    graph.progress = GraphProgress([node_progress("Polled", 66)], 0)
    wait_for(root, lambda: bar_text(frame.bars[0]) == "Polled: 66%")

    graph.progress = GraphProgress([], 0)
    wait_for(root, lambda: not frame.shown)
    assert not root.callback_errors


@pytest.mark.gui
def test_no_graph_is_fine(root):
    frame = QueryProgressFrame(root, lambda: None)
    frame.grid()
    frame.POLL_MS = 10
    pump(root, 0.2)
    assert not frame.shown and not root.callback_errors


@pytest.mark.gui
def test_destroy_stops_the_polling(root):
    # a timer left running after its widget is destroyed fails as a Tcl "background error"
    # ("invalid command name"), which doesn't go through report_callback_exception.
    background_errors = []
    root.createcommand("bgerror", lambda *args: background_errors.append(args))
    calls = []
    frame = QueryProgressFrame(root, lambda: calls.append(1))
    frame.grid()
    frame.POLL_MS = 10
    pump(root, 0.1)
    frame.destroy()
    count = len(calls)
    pump(root, 0.2)
    assert len(calls) == count
    assert not root.callback_errors and not background_errors


class BlockedPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.started = threading.Event()
        self.release = threading.Event()

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        self.started.set()
        assert self.release.wait(15)
        return ddbc.sql("select 1 as x")


@pytest.mark.gui
def test_main_window_shows_progress_of_running_node(root):
    mw = MainWindow(root, None)
    root.update()
    frame = mw.progress_frame
    frame.SHOW_AFTER = 0
    frame.POLL_MS = 20
    try:
        node = PipelineNode(name="Slow Node", plugin=BlockedPlugin())
        mw.graph.add_node(node)
        mw.graph.update()
        assert node.plugin.started.wait(15)

        wait_for(root, lambda: frame.shown and frame.winfo_ismapped())
        assert any("Slow Node" in bar_text(bar) for bar in visible_bars(frame))
        # it's in the log area, under the log and not on top of it
        assert frame.master is mw.logger_subframe
        assert frame.grid_info()["row"] == 1 and frame.grid_info()["columnspan"] == 2
        assert frame.winfo_y() >= mw.logger_subframe.text.winfo_y() + mw.logger_subframe.text.winfo_height()

        node.plugin.release.set()
        wait_for(root, lambda: not frame.shown)
        assert not frame.winfo_ismapped()
    finally:
        node.plugin.release.set()
        mw.graph.stop_thread(timeout=15)
    assert not root.callback_errors


@pytest.mark.gui
def test_main_window_copes_with_graph_being_replaced(root):
    mw = MainWindow(root, None)
    root.update()
    mw.progress_frame.POLL_MS = 10
    for _ in range(3):
        mw.config_new()
        pump(root, 0.1)
    assert not root.callback_errors
    mw.graph.stop_thread(timeout=15)
