import duckdb
import pandas as pd

from countess.plugins.regex import RegexToolPlugin

ddbc = duckdb.connect()

ddbc.from_df(
    pd.DataFrame(
        [
            {"stuff": "hello"},
            {"stuff": "backwards"},
            {"stuff": "noaardvark"},
        ]
    )
).create("n_0")

source = ddbc.table("n_0")


def test_tool_1():
    plugin = RegexToolPlugin()
    plugin.set_parameter("regex", ".*?([a]+).*")
    plugin.set_parameter("output.0.name", "foo")
    plugin.prepare(ddbc, source)
    plugin.set_parameter("column", "stuff")

    out = plugin.execute(ddbc, source)

    assert len(out) == 3
    assert out.columns == ["stuff", "foo"]
    assert sorted(out.fetchall()) == [("backwards", "a"), ("hello", ""), ("noaardvark", "aa")]


def test_tool_2():
    plugin = RegexToolPlugin()
    plugin.set_parameter("regex", ".*?([a]+).*")
    plugin.set_parameter("output.0.name", "foo")
    plugin.set_parameter("drop_column", True)
    plugin.prepare(ddbc, source)
    plugin.set_parameter("column", "stuff")

    out = plugin.execute(ddbc, source)

    assert len(out) == 3
    assert out.columns == ["foo"]
    assert sorted(out.fetchall()) == [("",), ("a",), ("aa",)]


def test_tool_3():
    plugin = RegexToolPlugin()
    plugin.set_parameter("regex", ".*?([a]+).*")
    plugin.set_parameter("output.0.name", "foo")
    plugin.set_parameter("drop_unmatch", True)
    plugin.prepare(ddbc, source)
    plugin.set_parameter("column", "stuff")

    out = plugin.execute(ddbc, source)

    assert len(out) == 2
    assert out.columns == ["stuff", "foo"]
    assert sorted(out.fetchall()) == [
        (
            "backwards",
            "a",
        ),
        (
            "noaardvark",
            "aa",
        ),
    ]


# ---- more of the regular expression tool


def test_dropping_the_only_column_leaves_just_the_extracted_ones():
    import duckdb  # pylint: disable=import-outside-toplevel

    from countess.plugins.regex import RegexToolPlugin  # pylint: disable=import-outside-toplevel

    ddbc = duckdb.connect()
    ddbc.sql("create table one as select * from (values ('ab12'), ('cd34'), ('zz')) v(txt)")
    source = ddbc.table("one")
    plugin = RegexToolPlugin()
    plugin.prepare(ddbc, source)
    plugin.set_parameter("column", "txt")
    plugin.set_parameter("regex", "([a-z]+)([0-9]+)")
    plugin.set_parameter("drop_column", True)
    plugin.set_parameter("output.0.name", "letters")
    plugin.set_parameter("output.0.datatype", "VARCHAR")
    plugin.set_parameter("output.1.name", "digits")
    plugin.set_parameter("output.1.datatype", "INTEGER")

    output = plugin.execute(ddbc, source)
    assert output.columns == ["letters", "digits"]
    assert sorted(output.fetchall(), key=repr) == [("", None), ("ab", 12), ("cd", 34)]  # (a non-match is blank)

    plugin.set_parameter("drop_unmatch", True)
    assert sorted(plugin.execute(ddbc, source).fetchall()) == [("ab", 12), ("cd", 34)]


def test_no_outputs_gives_the_table_back_with_or_without_the_column():
    import duckdb  # pylint: disable=import-outside-toplevel

    from countess.plugins.regex import RegexToolPlugin  # pylint: disable=import-outside-toplevel

    ddbc = duckdb.connect()
    ddbc.sql("create table two as select * from (values ('ab12', 1), ('zz', 2)) v(txt, n)")
    source = ddbc.table("two")
    plugin = RegexToolPlugin()
    plugin.prepare(ddbc, source)
    plugin.set_parameter("column", "txt")
    plugin.set_parameter("regex", "([a-z]+)([0-9]+)")
    assert plugin.execute(ddbc, source).columns == ["txt", "n"]  # (no output columns to extract into)
    plugin.set_parameter("drop_column", True)
    assert plugin.execute(ddbc, source).columns == ["n"]
