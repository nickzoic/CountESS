import logging
import os
import time
from decimal import Decimal

import duckdb
import pytest

from countess.core.cmd import run
from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.plugins.parquet import (
    PARQUET_COMPRESSION_CHOICES,
    PARQUET_FILE_TYPES,
    LoadParquetPlugin,
    SaveParquetPlugin,
)


@pytest.fixture(name="ddbc")
def fixture_ddbc():
    return duckdb.connect()


def write_parquet(ddbc, path, select, **options):
    """Make a Parquet file with DuckDB, independently of the plugin being tested."""
    clauses = "".join(f", {k} {v}" for k, v in options.items())
    ddbc.sql(f"COPY ({select}) TO '{path}' (FORMAT PARQUET{clauses})")


def load(ddbc, *filenames, row_limit=None, **params):
    plugin = LoadParquetPlugin()
    for num, filename in enumerate(filenames):
        plugin.set_parameter(f"files.{num}.filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.execute(ddbc, None, row_limit)


def save(ddbc, source, filename, row_limit=None, **params):
    plugin = SaveParquetPlugin()
    plugin.set_parameter("filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    plugin.prepare(ddbc, source)
    return plugin.execute(ddbc, source, row_limit)


def rows(ddbc, filename):
    return ddbc.read_parquet(str(filename)).fetchall()


def dtypes(ddbc, filename):
    rel = ddbc.read_parquet(str(filename))
    return dict(zip(rel.columns, (str(t) for t in rel.dtypes)))


# ---- loading


def test_load_parquet(ddbc, tmp_path):
    path = tmp_path / "in.parquet"
    write_parquet(ddbc, path, "select 1 as a, 2.5::double as b, 'three' as c union all select 4, 5.5, 'six'")
    output = load(ddbc, path)
    assert output.columns == ["a", "b", "c"]
    assert [str(t) for t in output.dtypes] == ["INTEGER", "DOUBLE", "VARCHAR"]  # types come from the file
    assert sorted(output.fetchall()) == [(1, 2.5, "three"), (4, 5.5, "six")]


def test_load_keeps_nulls_and_odd_column_names(ddbc, tmp_path):
    path = tmp_path / "in.parquet"
    write_parquet(ddbc, path, """select NULL::integer as "a b", 'x' as "quo""te", 'ünï' as "ünï cödé" """)
    output = load(ddbc, path)
    assert output.columns == ["a b", 'quo"te', "ünï cödé"]
    assert output.fetchall() == [(None, "x", "ünï")]


def test_load_with_no_files_gives_nothing(ddbc):
    assert LoadParquetPlugin().execute(ddbc, None) is None


def test_load_file_which_is_not_there_gives_nothing(ddbc, tmp_path):
    # (as for every loader which expands wildcards in filenames: no file matches, so no data)
    assert load(ddbc, tmp_path / "nothing.parquet") is None
    assert load(ddbc, tmp_path / "no*.parquet") is None


def test_load_something_which_is_not_parquet_is_an_error(ddbc, tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(duckdb.Error):
        load(ddbc, path).fetchall()


def test_filename_column(ddbc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_parquet(ddbc, "first.parquet", "select 1 as a")
    write_parquet(ddbc, "second.parquet", "select 2 as a")
    output = load(ddbc, "first.parquet", "second.parquet", filename_column=True)
    assert sorted(output.fetchall()) == [(1, "first.parquet"), (2, "second.parquet")]
    assert "filename" in output.columns


def test_load_several_files_matches_columns_by_name(ddbc, tmp_path):
    write_parquet(ddbc, tmp_path / "one.parquet", "select 1 as a, 'x' as b")
    write_parquet(ddbc, tmp_path / "two.parquet", "select 'y' as b, 2 as c")  # different columns, different order
    output = load(ddbc, tmp_path / "one.parquet", tmp_path / "two.parquet")
    assert sorted(output.columns) == ["a", "b", "c"]
    by_name = [dict(zip(output.columns, row)) for row in output.fetchall()]
    assert len(by_name) == 2
    assert {"a": 1, "b": "x", "c": None} in by_name
    assert {"a": None, "b": "y", "c": 2} in by_name


def test_load_files_matching_a_wildcard(ddbc, tmp_path):
    for n in range(3):
        write_parquet(ddbc, tmp_path / f"part{n}.parquet", f"select {n} as n")
    write_parquet(ddbc, tmp_path / "other.pq", "select 99 as n")
    output = load(ddbc, tmp_path / "part*.parquet")
    assert sorted(output.fetchall()) == [(0,), (1,), (2,)]


def test_row_limit_is_for_previews(ddbc, tmp_path):
    path = tmp_path / "many.parquet"
    write_parquet(ddbc, path, "select range as n from range(100000)")
    assert len(load(ddbc, path)) == 100000
    assert len(load(ddbc, path, row_limit=25)) == 25


def test_row_limit_is_shared_between_files(ddbc, tmp_path):
    for n in range(2):
        write_parquet(ddbc, tmp_path / f"f{n}.parquet", "select range as n from range(1000)")
    assert len(load(ddbc, tmp_path / "f0.parquet", tmp_path / "f1.parquet", row_limit=20)) == 20


WIDE_SELECT = """select
    true as bool, 1::tinyint as i8, 2::smallint as i16, 3::integer as i32, 4::bigint as i64, 6::utinyint as u8,
    7::ubigint as u64, 1.5::float as f32, 2.5::double as f64, 123.456::decimal(18,3) as dec,
    'text' as str, date '2024-01-02' as d, timestamp '2024-01-02 03:04:05' as ts,
    timestamptz '2024-01-02 03:04:05+00' as tstz, time '03:04:05' as t, interval 3 day as iv,
    [1,2,3] as lst, {'a': 1, 'b': 'x'} as st, 'abc'::blob as blb, NULL::integer as nul,
    'NaN'::double as nan, 'Infinity'::double as inf"""


def test_load_keeps_types(ddbc, tmp_path):
    path = tmp_path / "wide.parquet"
    write_parquet(ddbc, path, WIDE_SELECT)
    expected = dtypes(ddbc, path)
    output = load(ddbc, path)
    assert dict(zip(output.columns, (str(t) for t in output.dtypes))) == expected
    assert expected["dec"] == "DECIMAL(18,3)" and expected["lst"] == "INTEGER[]"
    row = dict(zip(output.columns, output.fetchone()))
    assert row["dec"] == Decimal("123.456") and row["lst"] == [1, 2, 3] and row["st"] == {"a": 1, "b": "x"}
    assert row["tstz"].utcoffset().total_seconds() == 0  # needs pytz to be installed


# ---- saving


@pytest.fixture(name="source")
def fixture_source(ddbc):
    ddbc.sql("create table n_0 as select * from (values (3, 'c', 3.5), (1, 'a', 1.5), (2, 'b', NULL)) t(n, s, f)")
    return ddbc.table("n_0")


def test_save_parquet(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    assert save(ddbc, source, path) is None
    assert dtypes(ddbc, path) == {"n": "INTEGER", "s": "VARCHAR", "f": "DECIMAL(2,1)"}
    assert sorted(rows(ddbc, path)) == [(1, "a", Decimal("1.5")), (2, "b", None), (3, "c", Decimal("3.5"))]


def test_save_makes_the_directory(ddbc, source, tmp_path):
    path = tmp_path / "does" / "not" / "exist" / "out.parquet"
    save(ddbc, source, path)
    assert len(rows(ddbc, path)) == 3


@pytest.mark.parametrize("compression", PARQUET_COMPRESSION_CHOICES)
def test_save_compression(ddbc, source, tmp_path, compression):
    path = tmp_path / "out.parquet"
    save(ddbc, source, path, compression=compression)
    used = {c for (c,) in ddbc.sql(f"select distinct compression from parquet_metadata('{path}')").fetchall()}
    assert used == {compression}
    assert len(rows(ddbc, path)) == 3


def test_default_compression_is_snappy_which_everything_reads(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    save(ddbc, source, path)
    assert {c for (c,) in ddbc.sql(f"select distinct compression from parquet_metadata('{path}')").fetchall()} == {
        "SNAPPY"
    }


def test_a_compression_which_does_not_exist_is_an_error(ddbc, source, tmp_path):
    with pytest.raises(duckdb.BinderException):
        save(ddbc, source, tmp_path / "out.parquet", compression="NOT_A_CODEC")
    assert not (tmp_path / "out.parquet").exists()


def test_save_sorted(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    save(ddbc, source, path, **{"sorting.0.order_by": "n"})
    assert [r[0] for r in rows(ddbc, path)] == [1, 2, 3]

    save(ddbc, source, path, **{"sorting.0.order_by": "n", "sorting.0.descending": True})
    assert [r[0] for r in rows(ddbc, path)] == [3, 2, 1]

    ddbc.sql("create table two_keys as select * from (values (1,'b'),(2,'a'),(1,'a'),(2,'b')) t(x, y)")
    save(
        ddbc,
        ddbc.table("two_keys"),
        path,
        **{"sorting.0.order_by": "x", "sorting.1.order_by": "y", "sorting.1.descending": True},
    )
    assert rows(ddbc, path) == [(1, "b"), (1, "a"), (2, "b"), (2, "a")]


def test_save_is_not_done_for_previews(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    save(ddbc, source, path, row_limit=10)
    assert not path.exists()  # a preview mustn't write (or overwrite) real files


def test_save_nothing_to_save_or_nowhere_to_save_it(ddbc, source, tmp_path):
    plugin = SaveParquetPlugin()
    plugin.set_parameter("filename", str(tmp_path / "out.parquet"))
    assert plugin.execute(ddbc, None) is None
    plugin = SaveParquetPlugin()  # no filename
    assert plugin.execute(ddbc, source) is None
    assert os.listdir(tmp_path) == []


def test_save_an_empty_table_keeps_the_columns(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    save(ddbc, source.filter("n > 100"), path)
    assert rows(ddbc, path) == []
    assert dtypes(ddbc, path) == {"n": "INTEGER", "s": "VARCHAR", "f": "DECIMAL(2,1)"}


def test_save_odd_column_names(ddbc, tmp_path):
    ddbc.sql("""create table odd as select 1 as "a b", 2 as "quo""te", 3 as "ünï" """)
    path = tmp_path / "out.parquet"
    save(ddbc, ddbc.table("odd"), path, **{"sorting.0.order_by": 'quo"te'})
    assert ddbc.read_parquet(str(path)).columns == ["a b", 'quo"te', "ünï"]


# ---- integers which are too big for Parquet


def test_sums_are_saved_as_integers_not_floats(ddbc, tmp_path, caplog):
    """DuckDB's SUM of integers is a HUGEINT, which Parquet can't store: DuckDB would write
    it as a DOUBLE, so a column of counts would come back as floating point."""
    ddbc.sql("create table t as select (range % 7)::bigint as k, range::bigint as n from range(1000)")
    totals = ddbc.sql("select k, sum(n) as total, count(*) as c from t group by k")
    assert [str(t) for t in totals.dtypes] == ["BIGINT", "HUGEINT", "BIGINT"]

    path = tmp_path / "out.parquet"
    with caplog.at_level(logging.INFO, logger="countess.plugins.parquet"):
        save(ddbc, totals, path, **{"sorting.0.order_by": "k"})
    assert dtypes(ddbc, path) == {"k": "BIGINT", "total": "BIGINT", "c": "BIGINT"}
    expected = [
        (k, sum(n for n in range(1000) if n % 7 == k), sum(1 for n in range(1000) if n % 7 == k)) for k in range(7)
    ]
    assert rows(ddbc, path) == expected
    assert "total" in caplog.text and "64 bit" in caplog.text

    # loading it back gives something which is still an integer
    assert [str(t) for t in load(ddbc, path).dtypes] == ["BIGINT", "BIGINT", "BIGINT"]


def test_unsigned_huge_integers_become_unsigned(ddbc, tmp_path):
    ddbc.sql("create table u as select 5::uhugeint as x, 6::bigint as y")
    path = tmp_path / "out.parquet"
    save(ddbc, ddbc.table("u"), path)
    assert dtypes(ddbc, path) == {"x": "UBIGINT", "y": "BIGINT"}


def test_a_number_which_does_not_fit_is_an_error_not_a_lie(ddbc, tmp_path):
    path = tmp_path / "out.parquet"
    big = ddbc.sql("select 170141183460469231731687303715884105727::hugeint as x")
    with pytest.raises(duckdb.ConversionException):
        save(ddbc, big, path)
    assert not path.exists()


def test_odd_column_names_survive_the_conversion(ddbc, tmp_path):
    """The columns are put through a projection when one of them has to be converted"""
    ddbc.sql(
        """create table odd as select 5::hugeint as "a b", 6 as "quo""te", 7::uhugeint as "ünï", 'x' as "sel ect" """
    )
    path = tmp_path / "out.parquet"
    save(ddbc, ddbc.table("odd"), path, **{"sorting.0.order_by": 'quo"te'})
    assert ddbc.read_parquet(str(path)).columns == ["a b", 'quo"te', "ünï", "sel ect"]
    assert dtypes(ddbc, path) == {"a b": "BIGINT", 'quo"te': "INTEGER", "ünï": "UBIGINT", "sel ect": "VARCHAR"}
    assert rows(ddbc, path) == [(5, 6, 7, "x")]


def test_columns_which_are_fine_are_left_alone(ddbc, source, tmp_path):
    path = tmp_path / "out.parquet"
    save(ddbc, source, path)
    assert dtypes(ddbc, path)["n"] == "INTEGER"  # (not widened, or anything)


# ---- what happens if saving goes wrong


def failing_source(ddbc):
    ddbc.sql("create or replace table big as select range::bigint as n from range(400000)")
    return ddbc.sql("select case when n = 350000 then error('boom') else n end as n from big")


def test_failure_leaves_the_existing_file_alone(ddbc, tmp_path):
    path = tmp_path / "out.parquet"
    write_parquet(ddbc, path, "select 1 as keep_me")
    original = path.read_bytes()
    with pytest.raises(duckdb.Error):
        save(ddbc, failing_source(ddbc), path)
    assert path.read_bytes() == original
    assert os.listdir(tmp_path) == ["out.parquet"]  # and no half written file left over


def test_failure_does_not_leave_a_partial_new_file(ddbc, tmp_path):
    with pytest.raises(duckdb.Error):
        save(ddbc, failing_source(ddbc), tmp_path / "new.parquet")
    assert os.listdir(tmp_path) == []


# ---- both together


def test_round_trip(ddbc, tmp_path):
    original = tmp_path / "original.parquet"
    write_parquet(ddbc, original, WIDE_SELECT)
    copy = tmp_path / "copy.parquet"
    save(ddbc, load(ddbc, original), copy)
    assert dtypes(ddbc, copy) == dtypes(ddbc, original)
    first, second = rows(ddbc, original), rows(ddbc, copy)
    assert repr(first) == repr(second)  # (repr so NaN == NaN)


def test_config_round_trip():
    for plugin_class in (LoadParquetPlugin, SaveParquetPlugin):
        plugin = plugin_class()
        params = list(plugin.get_parameters("", "."))
        copy = plugin_class()
        for key, value in params:
            copy.set_parameter(key, value, ".")
        assert list(copy.get_parameters("", ".")) == params
    assert dict(SaveParquetPlugin().get_parameters("", "."))["compression"] == "SNAPPY"


def test_file_types():
    assert PARQUET_FILE_TYPES == [("Parquet", [".parquet", ".pq"])]
    assert LoadParquetPlugin.file_types == SaveParquetPlugin.file_types == PARQUET_FILE_TYPES
    assert LoadParquetPlugin().files.file_types == PARQUET_FILE_TYPES


def test_documented():
    docs = open("docs/included-plugins/index.md", encoding="utf-8").read()
    for plugin_class, heading in ((LoadParquetPlugin, "Parquet Reader"), (SaveParquetPlugin, "Parquet Writer")):
        assert f"### {heading}" in docs
        assert plugin_class.link.endswith("#" + heading.lower().replace(" ", "-"))


def test_pipeline_from_a_config_file(tmp_path):
    """Load Parquet -> Group By -> Save Parquet, run the way `countess_cmd` does."""
    ddbc = duckdb.connect()
    write_parquet(
        ddbc, tmp_path / "input.parquet", "select (range % 3)::bigint as k, range::bigint as n from range(30)"
    )
    (tmp_path / "config.ini").write_text(
        """[Load]
_module = countess.plugins.parquet
_class = LoadParquetPlugin
_position = 100 100
files.0.filename = 'input.parquet'

[Group]
_module = countess.plugins.group_by
_class = GroupByPlugin
_parent.0 = Load
_position = 300 100
columns.0._label = 'k'
columns.0.index = True
columns.1._label = 'n'
columns.1.sum = True

[Save]
_module = countess.plugins.parquet
_class = SaveParquetPlugin
_parent.0 = Group
_position = 500 100
filename = 'output.parquet'
compression = 'ZSTD'
sorting.0.order_by = 'k'
""",
        encoding="utf-8",
    )
    run([str(tmp_path / "config.ini")])
    assert rows(ddbc, tmp_path / "output.parquet") == [(0, 135), (1, 145), (2, 155)]
    assert dtypes(ddbc, tmp_path / "output.parquet") == {"k": "BIGINT", "n__sum": "BIGINT"}  # (not DOUBLE)
    assert {
        c
        for (c,) in ddbc.sql(
            f"select distinct compression from parquet_metadata('{tmp_path}/output.parquet')"
        ).fetchall()
    } == {"ZSTD"}


# ---- in the GUI


@pytest.mark.gui
def test_previewing_a_parquet_file_with_every_kind_of_column(ddbc, tmp_path):
    """The preview has to cope with all the types which a Parquet file can have, not just
    the few which CSV loading gives."""
    from countess.gui.main import make_root  # pylint: disable=import-outside-toplevel
    from countess.gui.tabular import TabularDataFrame  # pylint: disable=import-outside-toplevel

    path = tmp_path / "wide.parquet"
    write_parquet(ddbc, path, WIDE_SELECT)

    graph = PipelineGraph()
    plugin = LoadParquetPlugin()
    plugin.set_parameter("files.0.filename", str(path))
    node = PipelineNode("load", plugin=plugin)
    graph.add_node(node)
    node.run(graph.thread_cursor)
    assert node.status == PipelineNodeStatus.RESULT, node.message

    root = make_root()
    root.geometry("1200x300")
    try:
        table = TabularDataFrame(root)
        table.pack(fill="both", expand=True)
        table.set_table(graph.ddbc, graph.ddbc.table(node.table_name))
        for column_number in range(len(table.columns)):  # sort by each kind of column in turn
            table.set_sort_order(column_number)
        deadline = time.time() + 5
        while time.time() < deadline:
            root.update()
            time.sleep(0.01)
        shown = [widget.get("1.0", "1.end") for widget in table.columns]
        assert len(shown) == len(plugin_columns(graph, node)) and all(shown)
    finally:
        root.destroy()
        graph.stop_thread(timeout=10)


def plugin_columns(graph, node):
    return graph.ddbc.table(node.table_name).columns


# ---- other software


def test_files_made_by_the_save_plugin_are_read_by_other_software(ddbc, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    ddbc.sql("create table t as select (range % 5)::bigint as k, range::bigint as n from range(1000)")
    totals = ddbc.sql("select k, sum(n) as total from t group by k")
    for compression in PARQUET_COMPRESSION_CHOICES:
        path = tmp_path / f"{compression}.parquet"
        save(ddbc, totals, path, compression=compression, **{"sorting.0.order_by": "k"})
        assert pq.read_table(path).num_rows == 5
        frame = pd.read_parquet(path)
        assert list(frame["k"]) == [0, 1, 2, 3, 4]
        assert str(frame["total"].dtype) == "int64"  # not float64, which is what a HUGEINT sum would have become
        assert list(frame["total"]) == [sum(n for n in range(1000) if n % 5 == k) for k in range(5)]


def test_files_made_by_other_software_are_loaded(ddbc, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import datetime  # pylint: disable=import-outside-toplevel

    utc = datetime.timezone.utc
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int32()),
            "when": pa.array([datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=utc)] * 3, pa.timestamp("us", tz="UTC")),
            "label": pa.array(["a", "b", "a"]).dictionary_encode(),
            "values": pa.array([[1, 2], [], None], pa.list_(pa.int64())),
            "point": pa.array(
                [{"x": 1.5, "y": 2.5}, {"x": 0.0, "y": None}, None],
                pa.struct([("x", pa.float64()), ("y", pa.float64())]),
            ),
            "maybe": pa.array([1.5, None, float("nan")], pa.float64()),
        }
    )
    path = tmp_path / "from_pyarrow.parquet"
    pq.write_table(table, path, compression="zstd")

    output = load(ddbc, path)
    assert [str(t) for t in output.dtypes] == [
        "INTEGER",
        "TIMESTAMP WITH TIME ZONE",
        "VARCHAR",
        "BIGINT[]",
        "STRUCT(x DOUBLE, y DOUBLE)",
        "DOUBLE",
    ]
    first, second, third = sorted(output.fetchall())
    assert first[0] == 1 and first[1] == datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=utc)
    assert (first[2], first[3], first[4], first[5]) == ("a", [1, 2], {"x": 1.5, "y": 2.5}, 1.5)
    assert (second[2], second[3], second[4], second[5]) == ("b", [], {"x": 0.0, "y": None}, None)
    assert third[2] == "a" and third[3] is None and third[4] is None
