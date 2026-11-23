"""What the preview pane shows for a node with no output rows: text, if its plugin has any."""

import time
import tkinter as tk
from unittest.mock import MagicMock

import pytest

from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin
from countess.gui.main import ConfiguratorWrapper, make_root
from countess.gui.tabular import TabularDataFrame
from countess.gui.widgets import TextPreviewFrame
from countess.plugins.csv import SaveCsvPlugin
from countess.plugins.parquet import SaveParquetPlugin


class RowsPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1
    show_preview = True

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        return ddbc.sql("select * from (values (1, 'x', 2.5), (2, 'y', NULL)) v(n, s, f)")


def descendants(widget):
    for w in widget.winfo_children():
        yield w
        yield from descendants(w)


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


@pytest.fixture(name="graph")
def fixture_graph():
    graph = PipelineGraph()
    yield graph
    graph.stop_thread(timeout=10)


def save_node(graph, plugin):
    parent = PipelineNode(name="rows", plugin=RowsPlugin())
    node = PipelineNode(name="save", plugin=plugin)
    graph.add_node(parent)
    graph.add_node(node)
    node.add_parent(parent)
    parent.run(graph.thread_cursor, 1000)
    node.run(graph.thread_cursor, 1000)
    return node


def show(root, graph, node):
    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    wrap.frame.pack(fill="both", expand=True)
    wait_for(root, lambda: wrap.preview_subframe is not None)
    root.update()
    return wrap


@pytest.mark.gui
def test_a_csv_save_node_shows_a_text_box_of_what_the_file_will_be(root, graph):
    plugin = SaveCsvPlugin()
    plugin.set_parameter("quoting", False)
    node = save_node(graph, plugin)
    assert node.status == PipelineNodeStatus.RESULT

    wrap = show(root, graph, node)

    preview = wrap.preview_subframe
    assert isinstance(preview, TextPreviewFrame)
    assert preview.text.get("1.0", "end-1c") == "n,s,f\n1,x,2.5\n2,y,\n"
    assert not [w for w in descendants(wrap.frame) if isinstance(w, TabularDataFrame)]  # (no table of rows)
    wrap.destroy()
    assert not root.callback_errors


@pytest.mark.gui
def test_the_text_box_is_for_looking_at_lines_of_a_file(root, graph):
    node = save_node(graph, SaveCsvPlugin())
    wrap = show(root, graph, node)
    text = wrap.preview_subframe.text

    assert str(text["state"]) == "disabled"  # (it can't be typed in)
    assert str(text["wrap"]) == "none"  # (lines are lines, not wrapped round)
    assert text["font"] == "TkFixedFont"
    scrollbars = [w for w in descendants(wrap.preview_subframe) if w.winfo_class() == "TScrollbar"]
    assert sorted(str(s["orient"]) for s in scrollbars) == ["horizontal", "vertical"]
    assert all(s.winfo_manager() == "grid" for s in scrollbars)  # (not just made, but there to be used)
    assert text["xscrollcommand"] and text["yscrollcommand"]  # (and what they scroll)
    wrap.destroy()


@pytest.mark.gui
def test_the_text_can_be_selected_and_copied(root, graph):
    node = save_node(graph, SaveCsvPlugin())
    wrap = show(root, graph, node)
    text = wrap.preview_subframe.text

    text.tag_add(tk.SEL, "1.0", "end-1c")

    assert text.selection_get().startswith('n,"s",f')
    wrap.destroy()


@pytest.mark.gui
def test_changing_a_setting_changes_the_text(root, graph):
    node = save_node(graph, SaveCsvPlugin())
    wrap = show(root, graph, node)
    first = wrap.preview_subframe
    assert first.text.get("1.0", "end-1c").startswith('n,"s",f\n')

    node.plugin.set_parameter("delimiter", ";")
    node.plugin.set_parameter("header", False)
    node.mark_dirty()
    node.run(graph.thread_cursor, 1000)
    wrap.update_preview_subframe()

    assert wrap.preview_subframe is not first
    assert wrap.preview_subframe.text.get("1.0", "end-1c") == '1;"x";2.5\n2;"y";\n'
    wrap.destroy()
    assert not root.callback_errors


@pytest.mark.gui
def test_a_node_which_hasnt_changed_keeps_its_text_box(root, graph):
    node = save_node(graph, SaveCsvPlugin())
    wrap = show(root, graph, node)
    first = wrap.preview_subframe

    wrap.update_preview_subframe()

    assert wrap.preview_subframe is first
    wrap.destroy()


@pytest.mark.gui
def test_a_node_which_has_rows_still_shows_them_as_a_table(root, graph):
    node = PipelineNode(name="rows", plugin=RowsPlugin())
    graph.add_node(node)
    node.run(graph.thread_cursor, 1000)

    wrap = show(root, graph, node)

    assert isinstance(wrap.preview_subframe, TabularDataFrame)
    wrap.destroy()


@pytest.mark.gui
def test_a_node_without_a_preview_still_has_no_preview_pane(root, graph, tmp_path):
    plugin = SaveParquetPlugin()
    plugin.set_parameter("filename", str(tmp_path / "out.parquet"))
    node = save_node(graph, plugin)

    wrap = ConfiguratorWrapper(root, node, graph.ddbc, MagicMock())
    wrap.frame.pack(fill="both", expand=True)
    for _ in range(100):
        root.update()
        time.sleep(0.01)

    assert wrap.preview_subframe is None
    wrap.destroy()


@pytest.mark.gui
def test_a_preview_which_went_wrong_shows_the_error_as_text(root, graph):
    plugin = SaveCsvPlugin()
    node = save_node(graph, plugin)
    node.plugin.sorted = MagicMock(side_effect=ValueError("could not sort"))  # (as if a sort went wrong)
    node.mark_dirty()
    node.run(graph.thread_cursor, 1000)
    assert node.status == PipelineNodeStatus.ERROR

    wrap = show(root, graph, node)

    texts = [w for w in descendants(wrap.preview_subframe) if isinstance(w, tk.Text)]
    assert texts and "could not sort" in texts[0].get("1.0", "end-1c")
    wrap.destroy()
