"""Tests for things which used to make the GUI slow: the log pane falling behind,
and previews being rebuilt (or built with far too many rows) needlessly."""

import logging
import logging.handlers
import queue
import time
from unittest.mock import MagicMock

import pytest

from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin
from countess.gui.main import ConfiguratorWrapper, LoggerFrame, make_root
from countess.gui.tabular import TabularDataFrame


def pump(root, seconds=0.2):
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


@pytest.fixture(name="root")
def fixture_root():
    root = make_root()
    root.geometry("900x600")
    yield root
    root.destroy()


# ---- log pane


@pytest.fixture(name="logger_frame")
def fixture_logger_frame(root):
    # A private queue: the main one is shared by every LoggerFrame in the process,
    # including any left behind by other tests.
    log_queue: queue.Queue = queue.Queue()
    frame = LoggerFrame(root, log_queue=log_queue)
    frame.pack(fill="both", expand=True)
    log = logging.getLogger("test_responsiveness")
    log.setLevel("INFO")
    log.propagate = False
    handler = logging.handlers.QueueHandler(log_queue)
    log.addHandler(handler)
    yield frame, log
    log.removeHandler(handler)


def text_lines(frame):
    return frame.text.get("1.0", "end-1c").splitlines()


@pytest.mark.gui
def test_log_pane_keeps_up_with_a_flood(root, logger_frame):
    frame, log = logger_frame
    # this used to take 100 messages a second, ie: 10 seconds for this.
    for i in range(1000):
        log.warning("message %d", i)
    start = time.time()
    wait_for(root, lambda: frame.count >= 1000)
    assert time.time() - start < 3
    assert text_lines(frame) == [f"message {i}" for i in range(1000)]


@pytest.mark.gui
def test_log_pane_does_not_hog_the_event_loop(root, logger_frame):
    frame, log = logger_frame
    for i in range(20000):
        log.warning("message %d", i)
    start = time.time()
    frame.logging_event()  # one pass of what runs every 100ms
    assert time.time() - start < 0.2
    assert 0 < frame.count < 20000  # ... which didn't try to do it all at once
    wait_for(root, lambda: frame.count >= 20000)


@pytest.mark.gui
def test_log_pane_keeps_a_limited_history(root, logger_frame):
    frame, log = logger_frame
    frame.LOGGING_MAX_LINES = 100
    for i in range(350):
        log.warning("message %d", i)
    wait_for(root, lambda: frame.count >= 350)
    lines = text_lines(frame)
    assert len(lines) == 100
    assert lines[-1] == "message 349" and lines[0] == "message 250"


@pytest.mark.gui
def test_progress_messages_go_to_progress_bars_not_the_log(root, logger_frame):
    frame, log = logger_frame
    log.warning("Node 1: 50%")
    log.warning("Node 2: 3/10")
    log.warning("something else")
    wait_for(root, lambda: frame.count >= 1)
    pump(root)
    assert set(frame.progress_bars) == {"Node 1", "Node 2"}
    assert text_lines(frame) == ["something else"]
    assert frame.count == 1


@pytest.mark.gui
def test_log_pane_only_follows_the_end_if_you_were_there(root, logger_frame):
    frame, log = logger_frame
    frame.LOGGING_MAX_LINES = 10000
    for i in range(500):
        log.warning("message %d", i)
    wait_for(root, lambda: frame.count >= 500)
    pump(root)
    assert frame.text.yview()[1] > 0.99  # was at the end, so stays at the end

    frame.text.yview_moveto(0)  # scroll back to read something
    pump(root)
    for i in range(500):
        log.warning("more %d", i)
    wait_for(root, lambda: frame.count >= 1000)
    pump(root)
    assert frame.text.yview()[0] == 0.0  # ... and the new messages didn't take us away


# ---- previews


class BigPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1
    show_preview = True

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        return ddbc.sql("select range as a, range * 2 as b, md5(range::varchar) as c from range(20000)")


@pytest.fixture(name="wrapper")
def fixture_wrapper(root):
    graph = PipelineGraph()
    node = PipelineNode(name="n", plugin=BigPlugin())
    graph.add_node(node)
    node.run(graph.thread_cursor)
    assert node.status == PipelineNodeStatus.RESULT
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    wrap.frame.pack(fill="both", expand=True)
    wait_for(root, lambda: wrap.preview_subframe is not None)
    pump(root, 0.3)
    yield wrap, node, graph
    wrap.destroy()
    graph.stop_thread(timeout=10)


@pytest.mark.gui
def test_reselecting_an_unchanged_node_keeps_its_preview(root, wrapper):
    wrap, node, _ = wrapper
    preview = wrap.preview_subframe
    for _ in range(5):
        wrap.notify()  # what selecting the node does
        root.update()
    assert wrap.preview_subframe is preview
    assert preview.winfo_exists()


@pytest.mark.gui
def test_changed_node_gets_a_new_preview(root, wrapper):
    wrap, node, graph = wrapper
    preview = wrap.preview_subframe
    node.mark_dirty()
    node.run(graph.thread_cursor)
    wrap.notify()
    assert wrap.preview_subframe is not None and wrap.preview_subframe is not preview
    assert not preview.winfo_exists()


@pytest.mark.gui
def test_change_while_preview_is_made_is_not_missed(root, wrapper):
    """If the node changes as the preview is being made it must be remade."""
    wrap, node, graph = wrapper
    wrap.hide_preview_subframe()
    node.mark_dirty()  # changes between the preview being asked for ...
    node.run(graph.thread_cursor)
    wrap.show_preview_subframe()
    preview = wrap.preview_subframe
    node.mark_dirty()  # ... and some time after it was made
    node.run(graph.thread_cursor)
    wrap.notify()
    assert wrap.preview_subframe is not preview


@pytest.mark.gui
def test_preview_does_not_start_from_a_thousand_rows(root, wrapper):
    wrap, node, _ = wrapper
    heights = []
    original = TabularDataFrame.refresh

    def recording_refresh(self, *a, **k):
        heights.append(self.height)
        return original(self, *a, **k)

    TabularDataFrame.refresh = recording_refresh
    try:
        wrap.replace_preview_subframe()
        pump(root, 0.5)
    finally:
        TabularDataFrame.refresh = original

    preview = wrap.preview_subframe
    assert heights and max(heights) < 200  # used to be 1000 for every build
    # ... but there are enough to fill the window, and no more than that.
    shown = preview.columns[0].get("1.0", "end-1c").count("\n")
    assert shown == preview.height
    visible = preview.columns[0].winfo_height() // preview._linespace
    assert preview.height >= visible - 1


@pytest.mark.gui
def test_resizing_makes_one_reset_not_one_per_event(root, wrapper):
    wrap, _, _ = wrapper
    preview = wrap.preview_subframe
    calls = []
    original = preview.refresh
    preview.refresh = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    for _ in range(30):
        preview._configure()  # pylint: disable=protected-access
    pump(root, 0.3)
    assert len(calls) <= 3  # the reset and the trimming it triggers


@pytest.mark.gui
def test_preview_columns_are_filled_in_one_call_each(root, wrapper):
    wrap, _, _ = wrapper
    preview = wrap.preview_subframe
    inserts = []
    original = type(preview.columns[0]).insert
    type(preview.columns[0]).insert = lambda self, *a, **k: (inserts.append(1), original(self, *a, **k))[1]
    try:
        preview.refresh()
    finally:
        type(preview.columns[0]).insert = original
    assert len(inserts) == len(preview.columns)
