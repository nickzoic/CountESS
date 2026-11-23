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
        assert text == 'a,b,"c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


def test_save_csv_gz():
    plugin = SaveCsvPlugin()
    plugin.set_parameter("header", True)
    plugin.set_parameter("filename", "tests/output1.csv.gz")
    source = ddbc.table("n_0")
    plugin.prepare(ddbc, source)
    assert plugin.execute(ddbc, source) is None

    with gzip.open("tests/output1.csv.gz", "rt") as fh:
        text = fh.read()
        assert text == 'a,b,"c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


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
        assert text == 'a,b,"c"\n1,2.01,"three"\n4,5.5,"six"\n7,8.8,"nine"\n'


# ---- more of the CSV plugins


import csv as csv_module
import io
import lzma

import pytest

from countess.plugins.csv import CSV_DELIMITER_CHOICES, csv_sql_type, write_to_csv
from countess.utils import compress


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


def save_csv_to(connection, path, source=None, **params):
    """Saves table t (or `source`) to `path`, where it isn't wanted back as text"""
    source = connection.table("t") if source is None else source
    plugin = SaveCsvPlugin()
    plugin.set_parameter("filename", str(path))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    plugin.prepare(connection, source)
    assert plugin.execute(connection, source) is None


def load_csv(connection, *filenames, **params):
    plugin = LoadCsvPlugin()
    for number, filename in enumerate(filenames):
        plugin.set_parameter(f"files.{number}.filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin, plugin.execute(connection, None)


@pytest.mark.parametrize("delimiter,sep", [(",", ","), (";", ";"), ("|", "|"), ("TAB", "\t"), ("SPACE", " ")])
def test_save_delimiters(csv_ddbc, tmp_path, delimiter, sep):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", delimiter=delimiter)
    lines = text.splitlines()
    assert lines[0] == sep.join(["n", '"s"', "f"])
    assert lines[1] == sep.join(["1", '"a b"', "2.5"])


def test_quoting_can_be_minimal(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", quoting=False)
    assert text == 'n,s,f\n1,a b,2.5\n2,"c,d",3.5\n3,"say ""hi""",\n'
    # (quoting the non-numbers always quotes the text, and the names of the columns it's in)
    assert save_csv(csv_ddbc, tmp_path / "out2.csv", quoting=True) == (
        'n,"s",f\n1,"a b",2.5\n2,"c,d",3.5\n3,"say ""hi""",\n'
    )


def test_quoting_everything(csv_ddbc, tmp_path):
    path = tmp_path / "all.csv"
    write_to_csv(csv_ddbc, csv_ddbc.table("t"), str(path), ",", csv_module.QUOTE_ALL)
    assert path.read_text(encoding="utf-8") == '"n","s","f"\n"1","a b","2.5"\n"2","c,d","3.5"\n"3","say ""hi""",\n'


# ---- quoting the non-numbers, which is done by DuckDB for every kind of file


def test_only_the_columns_which_are_not_numbers_are_quoted(csv_ddbc, tmp_path):
    csv_ddbc.sql(
        "create or replace table t as select 1 as i, 2.5 as d, true as b, 'x' as s, DATE '2024-01-02' as dt,"
        " TIMESTAMP '2024-01-02 03:04:05' as ts, 7::DECIMAL(5,2) as dc, 8::HUGEINT as h"
    )
    lines = save_csv(csv_ddbc, tmp_path / "out.csv").splitlines()
    assert lines == ['i,d,b,"s","dt","ts",dc,h', '1,2.5,true,"x","2024-01-02","2024-01-02 03:04:05",7.00,8']


def test_there_may_be_nothing_to_quote(csv_ddbc, tmp_path):
    csv_ddbc.sql("create or replace table t as select 1 as a, 2.5 as b")
    assert save_csv(csv_ddbc, tmp_path / "out.csv") == "a,b\n1,2.5\n"


def test_a_missing_value_is_blank_but_an_empty_string_is_quoted(csv_ddbc, tmp_path):
    csv_ddbc.sql("create or replace table t as select * from (values (1, ''), (2, NULL), (3, 'x')) v(n, s)")
    lines = save_csv(csv_ddbc, tmp_path / "out.csv").splitlines()
    assert lines[1:] == ['1,""', "2,", '3,"x"']


def test_awkward_text_comes_back_as_it_went(csv_ddbc, tmp_path):
    awkward = [
        "a,b",
        'say "hi"',
        "two\nlines",
        "tab\there",
        "semi;colon",
        "pipe|bar",
        "  padded  ",
        "caf\u00e9 \u2603",
        "'single'",
    ]
    csv_ddbc.create_function("awkward", lambda i: awkward[i], return_type=duckdb.sqltypes.DuckDBPyType("VARCHAR"))
    csv_ddbc.sql(f"create or replace table t as select range as n, awkward(range) as s from range({len(awkward)})")
    for delimiter, sep in ((",", ","), (";", ";"), ("|", "|"), ("TAB", "\t")):
        text = save_csv(csv_ddbc, tmp_path / f"{delimiter}.csv", delimiter=delimiter)
        rows = list(csv_module.reader(io.StringIO(text, newline=""), delimiter=sep))
        assert rows[0] == ["n", "s"]
        assert [row[1] for row in rows[1:]] == awkward, delimiter


def test_unusual_column_names_can_be_saved(csv_ddbc, tmp_path):
    csv_ddbc.sql('create or replace table t as select 1 as "a,b", \'x\' as "we""ird, name", \'y\' as "it\'s"')
    text = save_csv(csv_ddbc, tmp_path / "out.csv")
    rows = list(csv_module.reader(io.StringIO(text, newline="")))
    assert rows == [["a,b", 'we"ird, name', "it's"], ["1", "x", "y"]]


EXTENSIONS = ["", ".gz", ".bz2", ".xz"]
OPENERS = {"": open, ".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


def read_text(path):
    """What's in a file, decompressed if it's compressed, as the text which was written to it"""
    for extension, opener in OPENERS.items():
        if extension and str(path).lower().endswith(extension):
            with opener(path, "rt", encoding="utf-8", newline="") as fh:
                return fh.read()
    return path.read_text(encoding="utf-8")


def many_rows(connection, rows=20000):
    connection.sql(
        f"create or replace table t as select range as n, 'text ' || range as s, range * 0.25 as f from range({rows})"
    )


@pytest.fixture(name="small_blocks")
def fixture_small_blocks(monkeypatch):
    """So that even a small file is written as a lot of blocks, each of them a complete little file of its own"""
    monkeypatch.setattr(compress, "DEFAULT_BLOCK_SIZE", 4096)


@pytest.mark.parametrize("quoting", [True, False], ids=["quoting non-numbers", "minimal quoting"])
@pytest.mark.parametrize("header", [True, False])
def test_every_kind_of_file_has_exactly_the_same_text(csv_ddbc, tmp_path, quoting, header):
    texts = {}
    for extension in EXTENSIONS:
        path = tmp_path / f"out.csv{extension}"
        save_csv_to(csv_ddbc, path, quoting=quoting, header=header)
        texts[extension] = read_text(path)
    assert len(set(texts.values())) == 1, texts
    assert texts[""].count("\n") == (4 if header else 3)


@pytest.mark.parametrize("quoting", [True, False], ids=["quoting non-numbers", "minimal quoting"])
def test_every_kind_of_file_has_the_same_text_when_it_is_written_in_a_lot_of_blocks(
    csv_ddbc, tmp_path, small_blocks, quoting
):
    many_rows(csv_ddbc)
    texts = {}
    for extension in EXTENSIONS:
        path = tmp_path / f"out.csv{extension}"
        save_csv_to(csv_ddbc, path, quoting=quoting)
        texts[extension] = read_text(path)
    assert texts[""].count("\n") == 20001
    assert all(text == texts[""] for text in texts.values())


def test_quoting_everything_has_the_same_text_in_every_kind_of_file(csv_ddbc, tmp_path):
    texts = set()
    for extension in EXTENSIONS:
        path = tmp_path / f"all.csv{extension}"
        write_to_csv(csv_ddbc, csv_ddbc.table("t"), str(path), ",", csv_module.QUOTE_ALL)
        texts.add(read_text(path))
    assert texts == {'"n","s","f"\n"1","a b","2.5"\n"2","c,d","3.5"\n"3","say ""hi""",\n'}


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_sorting_works_for_every_kind_of_file(csv_ddbc, tmp_path, small_blocks, extension):
    many_rows(csv_ddbc, 3000)
    path = tmp_path / f"sorted.csv{extension}"
    save_csv_to(csv_ddbc, path, **{"sorting.0.order_by": "n", "sorting.0.descending": True})
    lines = read_text(path).splitlines()
    assert lines[:3] == ['n,"s",f', '2999,"text 2999",749.75', '2998,"text 2998",749.50']
    assert lines[-1] == '0,"text 0",0.00'


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_a_saved_file_loads_back_the_same(csv_ddbc, tmp_path, small_blocks, extension):
    many_rows(csv_ddbc)
    path = tmp_path / f"out.csv{extension}"
    save_csv_to(csv_ddbc, path)
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == csv_ddbc.table("t").fetchall()


@pytest.mark.parametrize("extension", [".gz", ".bz2", ".xz"])
def test_a_compressed_file_is_what_the_other_tools_can_read(csv_ddbc, tmp_path, small_blocks, extension):
    """(It's a lot of complete little files one after another, which is as good as one for them)"""
    many_rows(csv_ddbc, 5000)
    path = tmp_path / f"out.csv{extension}"
    save_csv_to(csv_ddbc, path)
    assert len(pd.read_csv(path)) == 5000
    if extension == ".gz":
        assert csv_ddbc.sql(f"select count(*), sum(n) from read_csv('{path}')").fetchone() == (5000, 12497500)


@pytest.mark.parametrize("extension", [".gz", ".bz2", ".xz"])
def test_a_file_with_no_rows_is_still_a_whole_compressed_file(csv_ddbc, tmp_path, extension):
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'a')) v(n, s) where false")
    path = tmp_path / f"empty.csv{extension}"
    save_csv_to(csv_ddbc, path)
    assert read_text(path) == 'n,"s"\n'
    save_csv_to(csv_ddbc, path, header=False)
    assert read_text(path) == ""


def test_the_extension_can_be_in_any_case(csv_ddbc, tmp_path):
    path = tmp_path / "OUT.CSV.GZ"
    save_csv_to(csv_ddbc, path)
    assert path.read_bytes()[:2] == b"\x1f\x8b"


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_it_isnt_done_by_pandas(csv_ddbc, tmp_path, monkeypatch, extension):
    def not_pandas(*args, **kwargs):
        raise AssertionError("pandas was used to write the file")

    monkeypatch.setattr(pd.DataFrame, "to_csv", not_pandas)
    for quoting in (True, False):
        path = tmp_path / f"out{quoting}.csv{extension}"
        save_csv_to(csv_ddbc, path, quoting=quoting)
        assert path.exists()


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_a_save_which_fails_leaves_nothing_where_there_was_nothing(csv_ddbc, tmp_path, extension):
    path = tmp_path / f"out.csv{extension}"
    with pytest.raises(duckdb.Error):
        save_csv_to(csv_ddbc, path, source=failing_source(csv_ddbc))
    assert list(tmp_path.iterdir()) == []  # (not the file and not a temporary file)


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_a_save_which_fails_leaves_the_old_file_as_it_was(csv_ddbc, tmp_path, extension):
    path = tmp_path / f"out.csv{extension}"
    path.write_bytes(b"the old file\n")
    with pytest.raises(duckdb.Error):
        save_csv_to(csv_ddbc, path, source=failing_source(csv_ddbc))
    assert path.read_bytes() == b"the old file\n"
    assert list(tmp_path.iterdir()) == [path]


def failing_source(connection):
    def explode(n: int) -> int:
        raise RuntimeError("boom")

    connection.create_function("explode", explode)
    connection.sql("create or replace view failing as select explode(n) as n from t")
    return connection.table("failing")


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_the_ways_of_quoting_duckdb_cant_do_are_written_by_pandas_to_every_kind_of_file(csv_ddbc, tmp_path, extension):
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'a'), (2, 'b')) v(n, s)")
    path = tmp_path / f"out.csv{extension}"
    write_to_csv(csv_ddbc, csv_ddbc.table("t"), str(path), ",", csv_module.QUOTE_NONE)
    assert read_text(path) == "n,s\n1,a\n2,b\n"


def test_save_xz(csv_ddbc, tmp_path):
    path = tmp_path / "out.csv.xz"
    save_csv_text = 'n,"s",f\n1,"a b",2.5\n2,"c,d",3.5\n3,"say ""hi""",\n'
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


# The loader is lenient about the files it reads, and these are the things that follows from that.  They
# are as intended, and described in the documentation of the CSV Reader: if one of them changes, so should it.


def test_doubled_quotes_inside_quoted_fields_are_kept_as_they_are(csv_ddbc, tmp_path):
    """A doubled quote inside a quoted field is not undone: it is loaded with both quotes still there."""
    path = tmp_path / "quoted.csv"
    path.write_text(QUOTED, encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == [(1, 'say ""hi""'), (2, "plain"), (3, "a,b")]


def test_text_with_quotes_in_it_is_not_the_same_after_being_saved_and_loaded(csv_ddbc, tmp_path):
    path = tmp_path / "out.csv"
    save_csv(csv_ddbc, path)  # (what's saved is right: say "hi" is written as "say ""hi""")
    assert '"say ""hi"""' in path.read_text(encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert 'say ""hi""' in [row[1] for row in output.fetchall()]  # ... but it is not read back that way


def test_a_quote_in_a_field_which_is_not_quoted_is_just_a_character(csv_ddbc, tmp_path):
    path = tmp_path / "bare.csv"
    path.write_text('a,b\nsay "hi",2\n', encoding="utf-8")
    _, output = load_csv(csv_ddbc, path, **{"columns.0.name": "a", "columns.0.type": "STRING"})
    assert output.fetchall() == [('say "hi"', "2")]


def test_commas_in_quoted_fields_are_fine(csv_ddbc, tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('n,s\n1,"a,b"\n2,"x;y"\n3,plain\n', encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert output.fetchall() == [(1, "a,b"), (2, "x;y"), (3, "plain")]


def test_newlines_in_quoted_fields_cannot_be_loaded(csv_ddbc, tmp_path):
    """The loader pads short rows (null_padding), which DuckDB's parallel reader can't do if a quoted
    field can have a newline in it, so a file which has one can't be loaded at all."""
    path = tmp_path / "quoted.csv"
    path.write_text('n,s\n1,"line one\nline two"\n2,plain\n', encoding="utf-8")
    with pytest.raises(duckdb.Error, match="CSV Error"):
        load_csv(csv_ddbc, path)[1].fetchall()  # (the relation is lazy, so it's when it's read that it fails)


def two_text_columns(connection, text, tmp_path, header=True):
    path = tmp_path / "two.csv"
    path.write_text(text, encoding="utf-8")
    return load_csv(
        connection,
        path,
        header=header,
        **{"columns.0.name": "a", "columns.0.type": "STRING", "columns.1.name": "b", "columns.1.type": "STRING"},
    )


def test_a_short_row_is_padded_with_empty_values(csv_ddbc, tmp_path):
    _, output = two_text_columns(csv_ddbc, "a,b\n1,2\n3\n4,5\n", tmp_path)
    assert output.fetchall() == [("1", "2"), ("3", None), ("4", "5")]


def test_a_row_with_too_many_fields_means_the_file_cant_be_loaded(csv_ddbc, tmp_path):
    with pytest.raises(duckdb.Error, match="sniffing"):
        two_text_columns(csv_ddbc, "a,b\n1,2\n3,4,5\n6,7\n", tmp_path)[1].fetchall()


def test_blank_lines_line_endings_and_a_byte_order_mark_are_all_fine(csv_ddbc, tmp_path):
    _, blank = two_text_columns(csv_ddbc, "a,b\n1,2\n\n3,4\n", tmp_path)
    assert blank.fetchall() == [("1", "2"), ("3", "4")]
    _, crlf = two_text_columns(csv_ddbc, "a,b\r\n1,2\r\n3,4\r\n", tmp_path)
    assert crlf.fetchall() == [("1", "2"), ("3", "4")]
    _, bom = two_text_columns(csv_ddbc, "\ufeffa,b\n1,2\n", tmp_path)
    assert bom.fetchall() == [("1", "2")]
    assert bom.columns == ["a", "b"]


def test_header_row_can_be_left_out(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", header=False)
    assert text.splitlines()[0] == '1,"a b",2.5'


def test_save_with_a_pipe_as_delimiter(csv_ddbc, tmp_path):
    text = save_csv(csv_ddbc, tmp_path / "out.csv", delimiter="|")
    assert text.splitlines()[1] == '1|"a b"|2.5'


def test_combining_nothing_gives_nothing(csv_ddbc):
    assert LoadCsvPlugin().combine(csv_ddbc, []) is None


@pytest.mark.parametrize("quoting", [True, False], ids=["quoting non-numbers", "minimal quoting"])
@pytest.mark.parametrize("header", [True, False])
def test_header_row_either_way_and_with_either_kind_of_quoting(csv_ddbc, tmp_path, quoting, header):
    # (both kinds of quoting are done by DuckDB: both have to do what they're told)
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'a'), (2, 'b')) v(n, s)")
    lines = save_csv(csv_ddbc, tmp_path / "out.csv", quoting=quoting, header=header).splitlines()
    assert len(lines) == (3 if header else 2)
    assert (lines[0] in ('n,"s"', "n,s")) == header
    assert lines[-1] in ('2,"b"', "2,b")


def test_header_row_left_out_when_quoting_everything(csv_ddbc, tmp_path):
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'a'), (2, 'b')) v(n, s)")
    for header, expected_lines in ((True, 3), (False, 2)):
        path = tmp_path / f"all{header}.csv"
        write_to_csv(csv_ddbc, csv_ddbc.table("t"), str(path), ",", csv_module.QUOTE_ALL, header=header)
        assert len(path.read_text(encoding="utf-8").splitlines()) == expected_lines


@pytest.mark.parametrize("header", [True, False])
def test_header_row_in_a_compressed_file(csv_ddbc, tmp_path, header):
    csv_ddbc.sql("create or replace table t as select * from (values (1, 'a'), (2, 'b')) v(n, s)")
    for extension, opener in ((".bz2", bz2.open), (".gz", gzip.open)):
        path = tmp_path / f"out.csv{extension}"
        plugin = SaveCsvPlugin()
        plugin.set_parameter("filename", str(path))
        plugin.set_parameter("header", header)
        source = csv_ddbc.table("t")
        plugin.prepare(csv_ddbc, source)
        plugin.execute(csv_ddbc, source)
        with opener(path, "rt", encoding="utf-8") as fh:
            assert len(fh.read().splitlines()) == (3 if header else 2), extension


def test_the_header_is_written_once_however_many_chunks_there_are(csv_ddbc, tmp_path):
    csv_ddbc.sql("create or replace table t as select range as n, 'x' as s from range(20000)")  # (several chunks)
    for header in (True, False):
        text = save_csv(csv_ddbc, tmp_path / f"big{header}.csv", header=header)
        lines = text.splitlines()
        assert len(lines) == 20000 + (1 if header else 0)
        assert sum(1 for line in lines if line.startswith("n,")) == (1 if header else 0)


def test_every_delimiter_which_is_offered_can_be_saved():
    plugin = SaveCsvPlugin()
    assert set(plugin.delimiter.choices) <= set(plugin.SEPARATORS)


# ---- INTEGER columns are 64 bit


def load_integers(connection, tmp_path, values, type_="INTEGER", filename="ints.csv"):
    path = tmp_path / filename
    path.write_text("n\n" + "".join(f"{v}\n" for v in values), encoding="utf-8")
    return load_csv(connection, path, **{"columns.0.name": "n", "columns.0.type": type_})[1]


def test_csv_sql_type():
    assert csv_sql_type("INTEGER") == "BIGINT"
    for other in ("STRING", "FLOAT", "BOOLEAN"):
        assert csv_sql_type(other) == other


def test_integer_columns_are_64_bit(csv_ddbc, tmp_path):
    values = [
        0,
        2147483647,
        2147483648,
        -2147483648,
        -2147483649,
        3000000000,
        9223372036854775807,
        -9223372036854775808,
    ]
    output = load_integers(csv_ddbc, tmp_path, values)
    assert [str(t) for t in output.dtypes] == ["BIGINT"]
    assert [row[0] for row in output.fetchall()] == values  # (none of them lost, which as 32 bit integers most were)


def test_integers_which_dont_fit_in_64_bits_are_empty(csv_ddbc, tmp_path):
    output = load_integers(
        csv_ddbc, tmp_path, ["9223372036854775808", "-9223372036854775809", "99999999999999999999", "5"]
    )
    assert [row[0] for row in output.fetchall()] == [None, None, None, 5]


def test_what_is_taken_for_an_integer(csv_ddbc, tmp_path):
    """DuckDB's rules: some things which aren't quite integers are, and what isn't a number is empty."""
    output = load_integers(csv_ddbc, tmp_path, ["+5", "007", " 12 ", "-3", "abc", '"1,000"', "NaN"])
    assert [row[0] for row in output.fetchall()] == [5, 7, 12, -3, None, None, None]


def test_a_number_with_a_fraction_is_rounded_to_an_integer_not_refused(csv_ddbc, tmp_path):
    """(This is DuckDB's cast, not something which is done here.)"""
    output = load_integers(csv_ddbc, tmp_path, ["1.4", "1.5", "2.0", "-1.5"])
    assert [row[0] for row in output.fetchall()] == [1, 2, 2, -2]


def test_integers_from_a_compressed_file_are_64_bit(csv_ddbc, tmp_path):
    path = tmp_path / "ints.csv.bz2"
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        fh.write("n\n3000000000\n-3000000000\n")
    _, output = load_csv(csv_ddbc, path, **{"columns.0.name": "n", "columns.0.type": "INTEGER"})
    assert [str(t) for t in output.dtypes] == ["BIGINT"]
    assert output.fetchall() == [(3000000000,), (-3000000000,)]


def test_an_integer_column_with_nothing_in_the_file_to_load_is_an_empty_one_of_the_same_type(csv_ddbc, tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("a\n1\n2\n", encoding="utf-8")
    _, output = load_csv(
        csv_ddbc,
        path,
        **{"columns.0.name": "a", "columns.0.type": "INTEGER", "columns.1.name": "extra", "columns.1.type": "INTEGER"},
    )
    assert output.columns == ["a", "extra"]
    assert [str(t) for t in output.dtypes] == ["BIGINT", "BIGINT"]
    assert output.fetchall() == [(1, None), (2, None)]


def test_columns_without_a_type_are_as_duckdb_finds_them_and_integers_are_64_bit_there_too(csv_ddbc, tmp_path):
    path = tmp_path / "sniffed.csv"
    path.write_text("a,b\n3000000000,x\n-3000000000,y\n", encoding="utf-8")
    _, output = load_csv(csv_ddbc, path)
    assert [str(t) for t in output.dtypes] == ["BIGINT", "VARCHAR"]
    assert output.fetchall() == [(3000000000, "x"), (-3000000000, "y")]


def test_big_integers_are_the_same_after_being_saved_and_loaded(csv_ddbc, tmp_path):
    csv_ddbc.sql(
        "create or replace table t as select * from (values (3000000000, 'a'), (-9223372036854775808, 'b')) v(n, s)"
    )
    path = tmp_path / "big.csv"
    save_csv(csv_ddbc, path)
    _, output = load_csv(
        csv_ddbc,
        path,
        **{"columns.0.name": "n", "columns.0.type": "INTEGER", "columns.1.name": "s", "columns.1.type": "STRING"},
    )
    assert sorted(output.fetchall()) == [(-9223372036854775808, "b"), (3000000000, "a")]


def test_other_types_are_not_changed_by_that(csv_ddbc, tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("s,b,i\nhello,true,7\n", encoding="utf-8")
    _, output = load_csv(
        csv_ddbc,
        path,
        **{
            "columns.0.name": "s", "columns.0.type": "STRING",
            "columns.1.name": "b", "columns.1.type": "BOOLEAN",
            "columns.2.name": "i", "columns.2.type": "INTEGER",
        },
    )  # fmt: skip
    assert [str(t) for t in output.dtypes] == ["VARCHAR", "BOOLEAN", "BIGINT"]
    assert output.fetchall() == [("hello", True, 7)]


def test_a_column_which_is_given_no_type_is_left_out(csv_ddbc, tmp_path):
    path = tmp_path / "three.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    _, output = load_csv(
        csv_ddbc,
        path,
        **{
            "columns.0.name": "a", "columns.0.type": "INTEGER",
            "columns.1.name": "b", "columns.1.type": "— NONE —",
            "columns.2.name": "c", "columns.2.type": "INTEGER",
        },
    )  # fmt: skip
    assert output.columns == ["a", "c"]
    assert output.fetchall() == [(1, 3), (4, 6)]


def test_integer_columns_keep_their_empty_values_and_stay_integers(csv_ddbc, tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text("n,m\n1,x\n,y\n3,z\n", encoding="utf-8")  # (a blank line would be skipped, so there's a comma)
    _, output = load_csv(
        csv_ddbc,
        path,
        **{"columns.0.name": "n", "columns.0.type": "INTEGER", "columns.1.name": "m", "columns.1.type": "STRING"},
    )
    assert [str(t) for t in output.dtypes] == ["BIGINT", "VARCHAR"]  # (not promoted to floating point)
    assert output.fetchall() == [(1, "x"), (None, "y"), (3, "z")]


def test_without_any_columns_listed_duckdb_works_out_the_types(csv_ddbc, tmp_path):
    path = tmp_path / "sniffed.csv"
    path.write_text(
        "i,x,d,t,b,s\n1,1.5,2024-01-02,2024-01-02 03:04:05,true,hello\n2,2.5,2024-02-03,2024-02-03 04:05:06,false,world\n",
        encoding="utf-8",
    )
    _, output = load_csv(csv_ddbc, path)
    assert dict(zip(output.columns, (str(t) for t in output.dtypes))) == {
        "i": "BIGINT",
        "x": "DOUBLE",
        "d": "DATE",
        "t": "TIMESTAMP",
        "b": "BOOLEAN",
        "s": "VARCHAR",
    }
