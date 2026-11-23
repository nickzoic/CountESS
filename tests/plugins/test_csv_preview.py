"""The preview of CSV Save, which is text, of what the file will look like, as it has no rows to show."""

import csv

import pytest

from countess.plugins.csv import CSV_PREVIEW_ROWS, SaveCsvPlugin
from countess.utils.compress import open_for_reading
from countess.utils.sql import SqlError


def make_plugin(ddbc, source, **params):
    plugin = SaveCsvPlugin()
    for key, value in params.items():
        plugin.set_parameter(key, value)
    plugin.prepare(ddbc, source)
    return plugin


def read_text(path):
    """The text of a file, as one which was compressed is when it's decompressed."""
    with open_for_reading(path) as fh:
        return fh.read().decode("utf-8")


@pytest.fixture(name="source")
def fixture_source(ddbc):
    ddbc.sql("create table t as select * from (values (1, 'x', 2.5), (3, 'z', NULL), (2, 'y', 0.5)) v(n, s, f)")
    return ddbc.table("t")


# ---- what it shows


def test_the_preview_is_the_file_as_it_will_be(ddbc, source):
    plugin = make_plugin(ddbc, source)
    # (strings and the names of their columns are quoted, numbers aren't, and a missing value is empty)
    assert plugin.preview_text(ddbc, source) == 'n,"s",f\n1,"x",2.5\n3,"z",\n2,"y",0.5\n'


def test_minimal_quoting(ddbc, source):
    plugin = make_plugin(ddbc, source, quoting=False)
    assert plugin.preview_text(ddbc, source) == "n,s,f\n1,x,2.5\n3,z,\n2,y,0.5\n"


def test_no_header_row(ddbc, source):
    plugin = make_plugin(ddbc, source, header=False, quoting=False)
    assert plugin.preview_text(ddbc, source) == "1,x,2.5\n3,z,\n2,y,0.5\n"


@pytest.mark.parametrize("delimiter,sep", [(",", ","), (";", ";"), ("|", "|"), ("TAB", "\t"), ("SPACE", " ")])
def test_the_delimiter(ddbc, source, delimiter, sep):
    plugin = make_plugin(ddbc, source, delimiter=delimiter, quoting=False)
    assert plugin.preview_text(ddbc, source).splitlines()[0] == sep.join(["n", "s", "f"])


def test_sorting(ddbc, source):
    plugin = make_plugin(ddbc, source, quoting=False)
    plugin.set_parameter("sorting.0.order_by", "n")
    assert plugin.preview_text(ddbc, source) == "n,s,f\n1,x,2.5\n2,y,0.5\n3,z,\n"
    plugin.set_parameter("sorting.0.descending", True)
    assert plugin.preview_text(ddbc, source) == "n,s,f\n3,z,\n2,y,0.5\n1,x,2.5\n"


def test_sorting_by_several_columns(ddbc):
    ddbc.sql("create table t as select * from (values (1, 'b'), (2, 'a'), (1, 'a'), (2, 'b')) v(n, s)")
    source = ddbc.table("t")
    plugin = make_plugin(ddbc, source, quoting=False)
    plugin.set_parameter("sorting.0.order_by", "n")
    plugin.set_parameter("sorting.0.descending", True)
    plugin.set_parameter("sorting.1.order_by", "s")
    assert plugin.preview_text(ddbc, source) == "n,s\n2,a\n2,b\n1,a\n1,b\n"


def test_a_sort_which_has_no_column_yet_is_an_error_which_says_so(ddbc, source):
    plugin = make_plugin(ddbc, source)
    plugin.set_parameter("sorting.0.order_by", "")
    with pytest.raises(SqlError):
        plugin.preview_text(ddbc, source)


def test_no_rows_is_the_header_and_nothing_else(ddbc):
    ddbc.sql("create table t as select * from (values (1, 'x')) v(n, s) where n > 1")
    source = ddbc.table("t")
    assert make_plugin(ddbc, source, quoting=False).preview_text(ddbc, source) == "n,s\n"
    assert make_plugin(ddbc, source, quoting=False, header=False).preview_text(ddbc, source) == ""


def test_it_is_there_before_a_filename_is_set(ddbc, source):
    plugin = make_plugin(ddbc, source, quoting=False)
    assert plugin.filename.value in (None, "")
    assert plugin.preview_text(ddbc, source).startswith("n,s,f\n")


def test_it_writes_nothing_where_the_file_is_to_go(ddbc, source, tmp_path):
    path = tmp_path / "dir" / "out.csv"
    plugin = make_plugin(ddbc, source, filename=str(path))
    plugin.preview_text(ddbc, source)
    assert not path.exists() and not path.parent.exists()
    assert plugin.execute(ddbc, source, row_limit=10) is None  # (a preview run doesn't write the file either)
    assert not path.exists()


# ---- it is the same as the file, whatever way the file is written


@pytest.fixture(name="awkward")
def fixture_awkward(ddbc):
    ddbc.sql("""create table awkward as select * from (values
            (1, 'plain', 2.5, true, NULL::VARCHAR, DATE '2024-01-02', 1.50::DECIMAL(5,2), 'ünï 日本'),
            (2, 'say "hi"', 3.0, false, 'x,y', DATE '2024-03-04', -0.25::DECIMAL(5,2), 'tab' || chr(9) || 'here'),
            (3, 'new' || chr(10) || 'line', NULL, NULL, '', NULL, NULL, ''),
            (4, 'semi;colon', 1e-7, true, 'it''s', DATE '1999-12-31', 100::DECIMAL(5,2), '|pipe|')
        ) v(n, s, f, b, z, d, dec, u)""")
    return ddbc.table("awkward")


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("extension", ["", ".gz", ".bz2", ".xz"])
@pytest.mark.parametrize("quoting", [True, False], ids=["quote non-numbers", "minimal quoting"])
@pytest.mark.parametrize("delimiter", [",", "TAB"])
def test_the_preview_is_exactly_what_is_written(ddbc, awkward, tmp_path, quoting, extension, header, delimiter):
    """(How booleans, nulls, numbers and the like come out depends on which of the ways of writing the file is
    used, which depends on the quoting and on whether it's compressed: so the preview has to use the same.)"""
    path = tmp_path / f"out.csv{extension}"
    plugin = make_plugin(ddbc, awkward, filename=str(path), quoting=quoting, header=header, delimiter=delimiter)

    preview = plugin.preview_text(ddbc, awkward)
    assert plugin.execute(ddbc, awkward) is None

    assert preview == read_text(path)


def test_what_is_written_for_booleans_missing_values_and_empty_strings(ddbc):
    """(As the writer says: booleans are true and false, a missing value is nothing whatever the column, and an
    empty string is a pair of quotes, with either kind of quoting.)"""
    ddbc.sql(
        "create table b as select * from (values (true, NULL::VARCHAR, ''), (false, 'x', 'y')) v(flag, maybe, empty)"
    )
    source = ddbc.table("b")
    assert make_plugin(ddbc, source, quoting=True).preview_text(ddbc, source) == (
        'flag,"maybe","empty"\ntrue,,""\nfalse,"x","y"\n'
    )
    assert make_plugin(ddbc, source, quoting=False).preview_text(ddbc, source) == (
        'flag,maybe,empty\ntrue,,""\nfalse,x,y\n'
    )


def test_a_compressed_file_is_previewed_as_text_and_not_as_it_will_be_stored(ddbc, source, tmp_path):
    plugin = make_plugin(ddbc, source, filename=str(tmp_path / "out.csv.bz2"), quoting=False)
    assert plugin.preview_text(ddbc, source) == "n,s,f\n1,x,2.5\n3,z,\n2,y,0.5\n"


# ---- when there are a lot of rows


def make_rows(ddbc, count):
    ddbc.sql(f"create or replace table big as select range as n, 'row ' || range as s from range({count})")
    return ddbc.table("big")


def test_only_the_start_is_shown_with_how_much_there_is_not_shown(ddbc):
    source = make_rows(ddbc, CSV_PREVIEW_ROWS + 150)
    lines = make_plugin(ddbc, source, quoting=False).preview_text(ddbc, source).splitlines()
    assert lines[0] == "n,s"
    assert len(lines) == 1 + CSV_PREVIEW_ROWS + 1  # the header, the rows, and the note
    assert lines[1] == "0,row 0" and lines[CSV_PREVIEW_ROWS] == f"{CSV_PREVIEW_ROWS - 1},row {CSV_PREVIEW_ROWS - 1}"
    assert lines[-1] == "… and 150 more rows"


def test_exactly_as_many_rows_as_are_shown_is_everything(ddbc):
    source = make_rows(ddbc, CSV_PREVIEW_ROWS)
    text = make_plugin(ddbc, source, quoting=False).preview_text(ddbc, source)
    assert len(text.splitlines()) == 1 + CSV_PREVIEW_ROWS
    assert "more row" not in text


def test_one_more_than_is_shown_says_so(ddbc):
    source = make_rows(ddbc, CSV_PREVIEW_ROWS + 1)
    assert make_plugin(ddbc, source, quoting=False).preview_text(ddbc, source).endswith("… and 1 more row\n")


def test_the_start_is_the_start_of_the_sorted_data(ddbc):
    source = make_rows(ddbc, 1000)
    plugin = make_plugin(ddbc, source, quoting=False)
    plugin.set_parameter("sorting.0.order_by", "n")
    plugin.set_parameter("sorting.0.descending", True)
    lines = plugin.preview_text(ddbc, source).splitlines()
    assert (
        lines[1] == "999,row 999"
        and lines[CSV_PREVIEW_ROWS] == f"{1000 - CSV_PREVIEW_ROWS},row {1000 - CSV_PREVIEW_ROWS}"
    )
    assert lines[-1] == f"… and {1000 - CSV_PREVIEW_ROWS} more rows"


def test_a_lot_of_rows_are_not_much_slower_than_a_few(ddbc):
    import time  # pylint: disable=import-outside-toplevel

    source = make_rows(ddbc, 2_000_000)
    plugin = make_plugin(ddbc, source)
    plugin.set_parameter("sorting.0.order_by", "n")
    start = time.time()
    plugin.preview_text(ddbc, source)
    assert time.time() - start < 5  # (a few tenths of a second: it's the top 100 of the sort, not the whole sort)


def test_the_quoting_constants_are_what_the_plugin_uses():
    assert SaveCsvPlugin.QUOTING == {False: csv.QUOTE_MINIMAL, True: csv.QUOTE_NONNUMERIC}
