"""A node whose plugin has no rows to show gets text to show in their place, if the plugin has any."""

from typing import Optional

import pytest

from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import DuckdbPlugin
from countess.plugins.csv import SaveCsvPlugin
from countess.plugins.parquet import SaveParquetPlugin


class RowsPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1

    def prepare_multi(self, ddbc, sources):
        pass

    def execute_multi(self, ddbc, sources, row_limit=None):
        return ddbc.sql("select * from (values (1, 'x'), (2, 'y')) v(n, s)")


class RowsWithPreviewPlugin(RowsPlugin):
    """(A plugin which has rows has no need of a preview text, so shouldn't be asked for one.)"""

    asked = False

    def preview_text_multi(self, ddbc, sources) -> Optional[str]:
        RowsWithPreviewPlugin.asked = True
        return "this shouldn't be used"


class BrokenPreviewPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 1
    num_outputs = 0

    def execute_multi(self, ddbc, sources, row_limit=None):
        return None

    def preview_text_multi(self, ddbc, sources) -> Optional[str]:
        raise ValueError("the preview went wrong")


@pytest.fixture(name="graph")
def fixture_graph():
    graph = PipelineGraph()
    yield graph
    graph.stop_thread(timeout=10)


def connect(graph, plugin):
    parent = PipelineNode(name="rows", plugin=RowsPlugin())
    child = PipelineNode(name="save", plugin=plugin)
    graph.add_node(parent)
    graph.add_node(child)
    child.add_parent(parent)
    parent.run(graph.thread_cursor, 1000)
    assert parent.status == PipelineNodeStatus.RESULT
    return parent, child


def test_a_node_which_saves_a_csv_file_has_a_preview_of_the_file(graph, tmp_path):
    _, child = connect(graph, SaveCsvPlugin())
    child.plugin.set_parameter("filename", str(tmp_path / "out.csv"))
    child.plugin.set_parameter("quoting", False)

    assert child.run(graph.thread_cursor, 1000) is False

    assert child.status == PipelineNodeStatus.RESULT
    assert child.message == ""
    assert child.preview_text == "n,s\n1,x\n2,y\n"
    assert not (tmp_path / "out.csv").exists()  # (it's a preview: nothing's saved)


def test_the_preview_follows_the_settings(graph):
    _, child = connect(graph, SaveCsvPlugin())
    child.plugin.set_parameter("quoting", False)
    child.run(graph.thread_cursor, 1000)
    assert child.preview_text == "n,s\n1,x\n2,y\n"

    child.plugin.set_parameter("delimiter", ";")
    child.plugin.set_parameter("header", False)
    child.mark_dirty()
    assert child.preview_text is None  # (not the old one while it's waiting to be done again)
    assert child.status == PipelineNodeStatus.DIRTY

    child.run(graph.thread_cursor, 1000)
    assert child.preview_text == "1;x\n2;y\n"


def test_a_node_which_has_no_preview_is_as_it_was(graph, tmp_path):
    """(No rows and no text is an error with nothing to say, which the interface doesn't show.)"""
    _, child = connect(graph, SaveParquetPlugin())
    child.plugin.set_parameter("filename", str(tmp_path / "out.parquet"))

    child.run(graph.thread_cursor, 1000)

    assert child.status == PipelineNodeStatus.ERROR
    assert child.message == ""
    assert child.preview_text is None


def test_a_node_which_has_rows_is_not_asked_for_text(graph):
    RowsWithPreviewPlugin.asked = False
    node = PipelineNode(name="rows", plugin=RowsWithPreviewPlugin())
    graph.add_node(node)

    node.run(graph.thread_cursor, 1000)

    assert node.status == PipelineNodeStatus.RESULT
    assert node.preview_text is None
    assert not RowsWithPreviewPlugin.asked


def test_a_preview_which_goes_wrong_is_the_nodes_error(graph):
    _, child = connect(graph, BrokenPreviewPlugin())

    child.run(graph.thread_cursor, 1000)

    assert child.status == PipelineNodeStatus.ERROR
    assert "the preview went wrong" in child.message
    assert child.preview_text is None


def test_a_run_which_is_out_of_date_doesnt_leave_its_text(graph):
    """(The same as for the status: a change made while it was running means what it made is not wanted.)"""
    _, child = connect(graph, SaveCsvPlugin())
    old_generation = child.generation
    child.mark_dirty()

    assert (
        child._settle(old_generation, PipelineNodeStatus.RESULT, "", "stale text") is True
    )  # pylint: disable=protected-access

    assert child.status == PipelineNodeStatus.DIRTY
    assert child.preview_text is None


def test_a_retry_leaves_no_text(graph):
    _, child = connect(graph, SaveCsvPlugin())
    child.run(graph.thread_cursor, 1000)
    assert child.preview_text is not None

    assert child._settle(child.generation, None) is True  # pylint: disable=protected-access

    assert child.preview_text is None
