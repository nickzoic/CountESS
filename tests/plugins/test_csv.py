import bz2
import gzip
import time

import duckdb
import pandas as pd

from countess.plugins.csv import LoadCsvPlugin, SaveCsvPlugin

ddbc = duckdb.connect()


def test_load_csv():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.csv")
    output = plugin.execute(ddbc, None)
    assert output.columns == ["thing", "count"]
    assert len(output) == 4


def test_load_csv_gz():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.csv.gz")
    output = plugin.execute(ddbc, None)
    assert output.columns == ["thing", "count"]
    assert len(output) == 4


def test_load_csv_bz2():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.csv.bz2")
    output = plugin.execute(ddbc, None)
    assert output.columns == ["thing", "count"]
    assert len(output) == 4


def test_load_tsv():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.tsv")
    plugin.set_parameter("delimiter", "TAB")
    output = plugin.execute(ddbc, None)
    assert output.columns == ["thing", "count"]
    assert len(output) == 4


def test_load_txt():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.txt")
    plugin.set_parameter("delimiter", "SPACE")
    output = plugin.execute(ddbc, None)
    assert output.columns == ["thing", "count"]
    assert len(output) == 4


def test_filename_column():
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", "tests/input1.csv")
    plugin.set_parameter("filename_column", True)
    output = plugin.execute(ddbc, None)
    assert "filename" in output.columns
    assert output.df()["filename"].iloc[1] == "tests/input1.csv"


df = pd.DataFrame([[1, 2.01, "three"], [4, 5.5, "six"], [7, 8.8, "nine"]], columns=["a", "b", "c"])
ddbc.from_df(df).create("n_0")


def test_save_csv():
    plugin = SaveCsvPlugin()
    plugin.set_parameter("header", True)
    plugin.set_parameter("filename", "tests/output1.csv")
    source = ddbc.table("n_0")
    plugin.prepare(ddbc, source)
    assert plugin.execute(ddbc, source) is None

    with open("tests/output1.csv", "r", encoding="utf-8") as fh:
        text = fh.read()
        assert text == '"a","b","c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


def test_save_csv_gz():
    plugin = SaveCsvPlugin()
    plugin.set_parameter("header", True)
    plugin.set_parameter("filename", "tests/output1.csv.gz")
    source = ddbc.table("n_0")
    plugin.prepare(ddbc, source)
    assert plugin.execute(ddbc, source) is None

    with gzip.open("tests/output1.csv.gz", "rt") as fh:
        text = fh.read()
        assert text == '"a","b","c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


def test_save_csv_bz2():
    plugin = SaveCsvPlugin()
    plugin.set_parameter("header", True)
    plugin.set_parameter("filename", "tests/output1.csv.bz2")
    source = ddbc.table("n_0")
    plugin.prepare(ddbc, source)
    assert plugin.execute(ddbc, source) is None
    time.sleep(0.5)

    with bz2.open("tests/output1.csv.bz2", "rt") as fh:
        text = fh.read()
        assert text == '"a","b","c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


# ---- more of the CSV plugins


import csv as csv_module
import lzma

import pytest

from countess.plugins.csv import CSV_DELIMITER_CHOICES, write_to_csv


@pytest.fixture(name="csv_ddbc")
def fixture_csv_ddbc():
    connection = duckdb.connect()
    connection.sql(
        "create table t as select * from (values (1, 'a b', 2.5), (2, 'c,d', 3.5), (3, 'say \"hi\"', NULL)) v(n, s, f)"
    )
    yield connection
    connection.close()


def save_csv(connection, path, **params):
    """Saves table t.  (Each save has a relation of its own, as each run of a pipeline does.)"""
    source = connection.table("t")
    plugin = SaveCsvPlugin()
    plugin.set_parameter("filename", str(path))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    plugin.prepare(connection, source)
    assert plugin.execute(connection, source) is None
    return path.read_text(encoding="utf-8")


def load_csv(connection, *filenames, **params):
    plugin = LoadCsvPlugin()
    for number, filename in enumerate(filenames):
        plugin.set_parameter(f"files.{number}.filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin, plugin.execute(connection, None)


@pytest.mark.parametrize("delimiter,sep", [(",", ","), (";", ";"), ("TAB", "\t"), ("SPACE", " ")])
def test_save_delimiters(csv_ddbc, tmp_path, delimiter, sep):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", delimiter=delimiter)
    lines = text.splitlines()
    assert lines[0] == sep.join(['"n"', '"s"', '"f"'])
    assert lines[1] == sep.join(["1", '"a b"', "2.5"])


def test_quoting_can_be_minimal(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", quoting=False)
    assert text == 'n,s,f\n1,a b,2.5\n2,"c,d",3.5\n3,"say ""hi""",\n'
    # (quoting the non-numbers is done by pandas, which also quotes a missing value: with minimal quoting it's blank)
    assert save_csv(csv_ddbc, tmp_path / "out2.csv", quoting=True) == (
        '"n","s","f"\n1,"a b",2.5\n2,"c,d",3.5\n3,"say ""hi""",""\n'
    )


def test_quoting_everything(csv_ddbc, tmp_path):
    path = tmp_path / "all.csv"
    write_to_csv(csv_ddbc.table("t"), str(path), ",", csv_module.QUOTE_ALL)
    assert path.read_text(encoding="utf-8") == '"n","s","f"\n"1","a b","2.5"\n"2","c,d","3.5"\n"3","say ""hi""",\n'


def test_save_xz(csv_ddbc, tmp_path):
    path = tmp_path / "out.csv.xz"
    save_csv_text = '"n","s","f"\n1,"a b",2.5\n2,"c,d",3.5\n3,"say ""hi""",""\n'
    plugin = SaveCsvPlugin()
    plugin.set_parameter("filename", str(path))
    source = csv_ddbc.table("t")
    plugin.prepare(csv_ddbc, source)
    plugin.execute(csv_ddbc, source)
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        assert fh.read() == save_csv_text


def test_load_xz(csv_ddbc, tmp_path):
    path = tmp_path / "in.csv.xz"
    with lzma.open(path, "wt", encoding="utf-8") as fh:
        fh.write("n,s\n1,x\n2,y\n")
    _, output = load_csv(csv_ddbc, path)
    assert output.columns == ["n", "s"]
    assert output.fetchall() == [(1, "x"), (2, "y")]


def test_save_sorted(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "up.csv", **{"sorting.0.order_by": "s"})
    assert [line.split(",")[0] for line in text.splitlines()[1:]] == ["1", "2", "3"]  # "a b", "c,d", 'say "hi"'
    text = save_csv(csv_ddbc, tmp_path / "down.csv", **{"sorting.0.order_by": "n", "sorting.0.descending": True})
    assert [line.split(",")[0] for line in text.splitlines()[1:]] == ["3", "2", "1"]


def test_save_sorted_by_several_columns(csv_ddbc, tmp_path):
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'b'), (2, 'a'), (1, 'a'), (2, 'b')) v(n, s)")
    text = save_csv(
        csv_ddbc,
        tmp_path / "out.csv",
        **{"sorting.0.order_by": "n", "sorting.1.order_by": "s", "sorting.1.descending": True},
    )
    assert text.splitlines()[1:] == ['1,"b"', '1,"a"', '2,"b"', '2,"a"']


def test_save_is_not_done_for_previews(csv_ddbc, tmp_path):
    path = tmp_path / "out.csv"
    plugin = SaveCsvPlugin()
    plugin.set_parameter("filename", str(path))
    source = csv_ddbc.table("t")
    plugin.prepare(csv_ddbc, source)
    plugin.execute(csv_ddbc, source, row_limit=10)
    assert not path.exists()


def test_save_makes_directories(csv_ddbc, tmp_path):
    path = tmp_path / "a" / "b" / "out.csv"
    save_csv(csv_ddbc, path)
    assert path.exists()


def test_nothing_to_save_or_nowhere_to_save_it(csv_ddbc, tmp_path):
    plugin = SaveCsvPlugin()
    assert plugin.execute(csv_ddbc, csv_ddbc.table("t")) is None  # no filename
    plugin.set_parameter("filename", str(tmp_path / "out.csv"))
    assert plugin.execute(csv_ddbc, None) is None
    assert list(tmp_path.iterdir()) == []


def test_loading_no_files_gives_nothing(csv_ddbc):
    assert LoadCsvPlugin().execute(csv_ddbc, None) is None


def test_every_delimiter_which_can_be_loaded(csv_ddbc, tmp_path):
    for name, sep in CSV_DELIMITER_CHOICES.items():
        path = tmp_path / f"in{ord(sep)}.csv"
        path.write_text(f"a{sep}b\n1{sep}2\n", encoding="utf-8")
        _, output = load_csv(csv_ddbc, path, delimiter=name)
        assert output.fetchall() == [(1, 2)], name


def test_round_trip(csv_ddbc, tmp_path):
    csv_ddbc.sql(
        "create or replace table t as select * from (values (1, 'a b', 2.5), (2, 'c,d', 3.5), (3, 'x', NULL)) v(n, s, f)"
    )
    path = tmp_path / "out.csv"
    save_csv(csv_ddbc, path, quoting=True)
    _, output = load_csv(csv_ddbc, path)
    assert sorted(output.fetchall(), key=repr) == sorted(csv_ddbc.table("t").fetchall(), key=repr)


QUOTED = 'n,s\n1,"say ""hi"""\n2,plain\n3,"a,b"\n'


@pytest.mark.xfail(
    strict=True,
    reason="the loader asks DuckDB for strict_mode=False, and in that mode the doubled quotes which stand for a quote "
    'inside a quoted field are not undone: "say ""hi""" is loaded as say ""hi"" (DuckDB\'s default does it right)',
)
def test_quotes_inside_quoted_fields(csv_ddbc, tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text(QUOTED, encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == [(1, 'say "hi"'), (2, "plain"), (3, "a,b")]


@pytest.mark.xfail(
    strict=True,
    reason="the same thing, so text with a quote in it isn't the same after being saved and loaded",
)
def test_round_trip_of_text_with_quotes(csv_ddbc, tmp_path):
    path = tmp_path / "out.csv"
    save_csv(csv_ddbc, path)  # (the saved file is right: say "hi" is written as "say ""hi""")
    assert '"say ""hi"""' in path.read_text(encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert 'say "hi"' in [row[1] for row in output.fetchall()]


def test_commas_in_quoted_fields_are_fine(csv_ddbc, tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('n,s\n1,"a,b"\n2,"x;y"\n3,plain\n', encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == [(1, "a,b"), (2, "x;y"), (3, "plain")]


@pytest.mark.xfail(
    strict=True,
    raises=duckdb.Error,
    reason="the loader asks DuckDB for null_padding, which its parallel reader can't use with a newline inside a "
    "quoted field: 'The parallel scanner does not support null_padding in conjunction with quoted new lines'",
)
def test_newlines_in_quoted_fields(csv_ddbc, tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('n,s\n1,"line one\nline two"\n2,plain\n', encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == [(1, "line one\nline two"), (2, "plain")]


@pytest.mark.xfail(
    strict=True,
    reason='the "CSV header row?" setting is never passed on to the code which writes the file, so there is '
    "always a header row",
)
def test_header_row_can_be_left_out(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", header=False)
    assert text.splitlines()[0] == '1,"a b",2.5'


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="| is one of the choices of delimiter, but it's not in the table which looks the delimiters up when saving "
    "(it is when loading)",
)
def test_save_with_a_pipe_as_delimiter(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", delimiter="|")
    assert text.splitlines()[1] == '1|"a b"|2.5'


def test_combining_nothing_gives_nothing(csv_ddbc):
    assert LoadCsvPlugin().combine(csv_ddbc, []) is None
