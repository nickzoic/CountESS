import duckdb
import numpy as np
import pandas as pd
import pytest

ddbc = duckdb.connect()

from countess.plugins.join import JoinPlugin

ddbc.from_df(
    pd.DataFrame(
        [
            {"foo": 1, "bar": 2, "baz": 3},
            {"foo": 4, "bar": 5, "baz": 6},
            {"foo": 7, "bar": 8, "baz": 9},
        ]
    )
).create("n_1")

ddbc.from_df(
    pd.DataFrame(
        [
            {"foo": 1, "qux": "a"},
            {"foo": 4, "qux": "b"},
            {"foo": 8, "qux": "c"},
        ]
    )
).create("n_2")

sources = {"one": ddbc.table("n_1"), "two": ddbc.table("n_2")}


def test_join_inner():
    plugin = JoinPlugin()

    plugin.set_parameter("inputs.0.join_on", "foo")
    plugin.set_parameter("inputs.0.required", True)
    plugin.set_parameter("inputs.1.join_on", "foo")
    plugin.set_parameter("inputs.1.required", True)
    plugin.prepare_multi(ddbc, sources)

    output = plugin.execute_multi(ddbc, sources)

    assert output.columns == ["foo", "bar", "baz", "qux"]
    assert output.df().equals(
        pd.DataFrame(
            [
                {"foo": 1, "bar": 2, "baz": 3, "qux": "a"},
                {"foo": 4, "bar": 5, "baz": 6, "qux": "b"},
            ]
        )
    )


def test_join_left():
    plugin = JoinPlugin()

    plugin.set_parameter("inputs.0.join_on", "foo")
    plugin.set_parameter("inputs.0.required", True)
    plugin.set_parameter("inputs.1.join_on", "foo")
    plugin.set_parameter("inputs.1.required", False)
    plugin.prepare_multi(ddbc, sources)

    output = plugin.execute_multi(ddbc, sources)

    assert output.columns == ["foo", "bar", "baz", "qux"]
    assert output.df().equals(
        pd.DataFrame(
            [
                {"foo": 1, "bar": 2, "baz": 3, "qux": "a"},
                {"foo": 4, "bar": 5, "baz": 6, "qux": "b"},
                {"foo": 7, "bar": 8, "baz": 9, "qux": None},
            ]
        )
    )


def test_join_right():
    plugin = JoinPlugin()

    plugin.set_parameter("inputs.0.join_on", "foo")
    plugin.set_parameter("inputs.0.required", False)
    plugin.set_parameter("inputs.1.join_on", "foo")
    plugin.set_parameter("inputs.1.required", True)
    plugin.prepare_multi(ddbc, sources)

    output = plugin.execute_multi(ddbc, sources)

    assert output.columns == ["foo", "bar", "baz", "qux"]
    assert output.fetchall() == [(1, 2, 3, "a"), (4, 5, 6, "b"), (None, None, None, "c")]


def test_join_full():
    plugin = JoinPlugin()

    plugin.set_parameter("inputs.0.join_on", "foo")
    plugin.set_parameter("inputs.0.required", False)
    plugin.set_parameter("inputs.1.join_on", "foo")
    plugin.set_parameter("inputs.1.required", False)
    plugin.prepare_multi(ddbc, sources)

    output = plugin.execute_multi(ddbc, sources)
    assert output.columns == ["foo", "bar", "baz", "qux"]
    assert output.fetchall() == [(1, 2, 3, "a"), (4, 5, 6, "b"), (None, None, None, "c"), (7, 8, 9, None)]


# ---- more of the join plugin

import logging

import duckdb
import pytest

from countess.plugins.join import _join_how


@pytest.fixture(name="join_ddbc")
def fixture_join_ddbc():
    connection = duckdb.connect()
    connection.sql("create table a as select * from (values (1, 'x', 10), (2, 'y', 20), (3, 'z', 30)) v(k1, k2, va)")
    connection.sql("create table b as select * from (values (1, 'x', 100), (2, 'q', 200), (4, 'z', 400)) v(j1, j2, vb)")
    connection.sql("create table d as select * from (values (1, 7), (3, 9)) v(m1, vd)")
    connection.sql(
        "create table e as select * from (values (1, 'x', 10, 'first'), (2, 'y', 99, 'second')) v(n1, n2, n3, ve)"
    )
    yield connection
    connection.close()


def run_join(connection, tables, settings, row_limit=None):
    """`tables` are the names of tables, `settings` maps parameter names (like "0.join_on") to values"""
    sources = {name: connection.table(name) for name in tables}
    plugin = JoinPlugin()
    for key, value in settings.items():
        plugin.set_parameter(f"inputs.{key}", value)
    plugin.prepare_multi(connection, sources)
    return plugin, plugin.execute_multi(connection, sources, row_limit)


def test_how_to_join():
    assert _join_how(True, True) == "INNER"
    assert _join_how(True, False) == "LEFT OUTER"
    assert _join_how(False, True) == "RIGHT OUTER"
    assert _join_how(False, False) == "FULL OUTER"


def test_join_on_two_columns(join_ddbc, assert_rows):
    _, output = run_join(
        join_ddbc, ["a", "b"], {"0.join_on": "k1", "0.join_2": "k2", "1.join_on": "j1", "1.join_2": "j2"}
    )
    assert output.columns == ["k1", "k2", "va", "j1", "j2", "vb"]
    assert_rows(output, [(1, "x", 10, 1, "x", 100)])  # (2 matches on the first column but not the second)


def test_join_on_three_columns(join_ddbc, assert_rows):
    settings = {
        "0.join_on": "k1",
        "0.join_2": "k2",
        "0.join_3": "va",
        "1.join_on": "n1",
        "1.join_2": "n2",
        "1.join_3": "n3",
    }
    _, output = run_join(join_ddbc, ["a", "e"], settings)
    assert_rows(output, [(1, "x", 10, 1, "x", 10, "first")])  # (2 matches the first two but not the third)


def test_a_second_column_to_join_on_is_only_used_when_both_have_one(join_ddbc, assert_rows):
    _, output = run_join(join_ddbc, ["a", "b"], {"0.join_on": "k1", "0.join_2": "k2", "1.join_on": "j1"})
    assert_rows(output, [(1, "x", 10, 1, "x", 100), (2, "y", 20, 2, "q", 200)])


def test_three_tables(join_ddbc, assert_rows):
    _, output = run_join(join_ddbc, ["a", "b", "d"], {"0.join_on": "k1", "1.join_on": "j1", "2.join_on": "m1"})
    assert output.columns == ["k1", "k2", "va", "j1", "j2", "vb", "m1", "vd"]
    assert_rows(output, [(1, "x", 10, 1, "x", 100, 1, 7)])  # (only 1 is in all three)


def test_three_tables_with_the_first_required_and_the_others_not(join_ddbc, assert_rows):
    settings = {
        "0.join_on": "k1",
        "0.required": True,
        "1.join_on": "j1",
        "1.required": False,
        "2.join_on": "m1",
        "2.required": False,
    }
    _, output = run_join(join_ddbc, ["a", "b", "d"], settings)
    assert_rows(
        output,
        [(1, "x", 10, 1, "x", 100, 1, 7), (2, "y", 20, 2, "q", 200, None, None), (3, "z", 30, None, None, None, 3, 9)],
    )


def test_a_join_column_can_be_dropped(join_ddbc, assert_rows):
    _, output = run_join(join_ddbc, ["a", "b"], {"0.join_on": "k1", "1.join_on": "j1", "1.drop": True})
    assert output.columns == ["k1", "k2", "va", "j2", "vb"]
    assert_rows(output, [(1, "x", 10, "x", 100), (2, "y", 20, "q", 200)])
    _, first = run_join(join_ddbc, ["a", "b"], {"0.join_on": "k1", "0.drop": True, "1.join_on": "j1"})
    assert first.columns == ["k2", "va", "j1", "j2", "vb"]


def test_columns_with_the_same_name_are_both_kept_with_one_renamed(join_ddbc):
    join_ddbc.sql("create table s as select * from (values (1, 'p'), (2, 'q')) v(k1, same)")
    join_ddbc.sql("create table s2 as select * from (values (1, 'P'), (2, 'Q')) v(k1, same)")
    _, output = run_join(join_ddbc, ["s", "s2"], {"0.join_on": "k1", "1.join_on": "k1"})
    assert output.columns == ["k1", "same", "same"]  # (the join key is only there once)
    output.to_table("joined")  # ... and DuckDB renames one of them when it's made into a table
    assert join_ddbc.table("joined").columns == ["k1", "same", "same_1"]
    assert sorted(join_ddbc.table("joined").fetchall()) == [(1, "p", "P"), (2, "q", "Q")]


def test_a_missing_key_never_matches(join_ddbc, assert_rows):
    join_ddbc.sql("create table n1 as select * from (values (1, 'a'), (NULL, 'b')) v(k, x)")
    join_ddbc.sql("create table n2 as select * from (values (1, 'c'), (NULL, 'd')) v(k, y)")
    _, inner = run_join(join_ddbc, ["n1", "n2"], {"0.join_on": "k", "1.join_on": "k"})
    assert_rows(inner, [(1, "a", "c")])
    _, full = run_join(
        join_ddbc, ["n1", "n2"], {"0.join_on": "k", "0.required": False, "1.join_on": "k", "1.required": False}
    )
    assert_rows(full, [(1, "a", "c"), (None, "b", None), (None, None, "d")])  # (the NULLs are just left unmatched)


def test_row_limit(join_ddbc):
    _, output = run_join(join_ddbc, ["a", "b"], {"0.join_on": "k1", "1.join_on": "j1"}, row_limit=1)
    assert len(output.fetchall()) == 1


def test_an_empty_table(join_ddbc, assert_rows):
    join_ddbc.sql("create table nothing as select * from (values (1, 5)) v(k1, vq) where false")
    _, inner = run_join(join_ddbc, ["a", "nothing"], {"0.join_on": "k1", "1.join_on": "k1"})
    assert inner.columns == ["k1", "k2", "va", "vq"]
    assert inner.fetchall() == []
    _, left = run_join(join_ddbc, ["a", "nothing"], {"0.join_on": "k1", "1.join_on": "k1", "1.required": False})
    assert_rows(left, [(1, "x", 10, None), (2, "y", 20, None), (3, "z", 30, None)])


def test_the_inputs_follow_the_number_of_tables(join_ddbc):
    plugin, _ = run_join(join_ddbc, ["a", "b", "d"], {"0.join_on": "k1", "1.join_on": "j1", "2.join_on": "m1"})
    assert [row.label for row in plugin.inputs] == ["Input 1: a", "Input 2: b", "Input 3: d"]
    assert plugin.inputs[1].join_on.choices == ["j1", "j2", "vb"]

    two = {"a": join_ddbc.table("a"), "b": join_ddbc.table("b")}
    plugin.execute_multi(join_ddbc, two)  # (one table fewer)
    assert [row.label for row in plugin.inputs] == ["Input 1: a", "Input 2: b"]
    plugin.execute_multi(join_ddbc, {**two, "d": join_ddbc.table("d")})  # (and back again)
    assert len(plugin.inputs) == 3


def test_nothing_to_join(join_ddbc):
    plugin = JoinPlugin()
    assert plugin.execute_multi(join_ddbc, {}) is None
    assert plugin.execute_multi(join_ddbc, {"a": join_ddbc.table("a")}) is None  # (one table can't be joined)
    assert plugin.execute_multi(join_ddbc, {"a": join_ddbc.table("a"), "b": None}) is None


def test_a_join_between_things_which_cant_be_compared_gives_nothing(join_ddbc, caplog):
    join_ddbc.sql("create table words as select * from (values ('abc', 1), ('2', 2)) v(k1, vw)")
    with caplog.at_level(logging.INFO):
        _, output = run_join(join_ddbc, ["a", "words"], {"0.join_on": "k1", "1.join_on": "k1"})
    assert output is None  # (a number can be compared with '2' but not with 'abc')
    assert "abc" in caplog.text
