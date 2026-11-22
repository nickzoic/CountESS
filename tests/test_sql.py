import datetime
import decimal
import math
import random

import duckdb
import numpy as np
import pytest

from countess.utils.sql import (
    EMPTY,
    Fn,
    Fragment,
    Id,
    Lit,
    Raw,
    Sql,
    Type,
    columns_alias,
    join,
    regex_escape,
)

AWKWARD_NAMES = [
    "plain",
    "a b",
    'q"uote',
    "a.b",
    "select",
    "ünï 日本",
    "new\nline",
    "(x)",
    '"',
    ".",
    "a.b.c",
    "Count",
    "x" * 300,
    "'",
    "\\",
    "%s",
    "{x}",
]


@pytest.fixture(name="ddbc")
def fixture_ddbc():
    connection = duckdb.connect()
    yield connection
    connection.close()


# ---- identifiers


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_an_identifier_is_the_name_of_a_column_whatever_it_is(ddbc, name):
    ddbc.sql(str(Sql.template("create table t as select 41 as {col}", col=Id(name))))
    assert ddbc.table("t").columns == [name]
    assert ddbc.sql(str(Sql.template("select {col} + 1 from t", col=Id(name)))).fetchone() == (42,)


@pytest.mark.parametrize("name,error", [(None, TypeError), (5, TypeError), ("", ValueError), ("a\x00b", ValueError)])
def test_things_which_cant_be_identifiers_are_refused_where_they_are_made(name, error):
    with pytest.raises(error):
        Id(name)


def test_dotted_names():
    assert str(Id.dotted("my table", "a.b")) == '"my table"."a.b"'


# ---- values

ROUND_TRIPS = [
    "plain", "it's", "back\\slash", "new\nline", "ünï 日本", "", "%", "'; drop table t; --", 'q"uote',
    0, 7, -7, 2**70, -(2**70),
    1.5, -1.5, 1e-20, 1e30, 0.1, -0.0,
    True, False, None,
    decimal.Decimal("1.50"), decimal.Decimal("-0.001"), decimal.Decimal("12345678901234567890.5"), decimal.Decimal("100"),
    datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 2, 3, 4, 5, 678901), datetime.time(3, 4, 5),
    b"\x00\x01bytes\xff",
    [1, 2, 3], ["a", "b"], [], (1, 2),
]  # fmt: skip


@pytest.mark.parametrize("value", ROUND_TRIPS, ids=repr)
def test_a_value_is_the_same_value_when_DuckDB_gives_it_back(ddbc, value):
    got = ddbc.sql(f"select {Lit(value)}").fetchone()[0]
    assert got == (list(value) if isinstance(value, tuple) else value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_values_and_not_names_of_columns(ddbc, value):
    """(nan::DOUBLE, which is what was written, is a column called nan, and a column which is called that gets used)"""
    ddbc.sql("create table t as select 12345.0 as nan, 6.0 as inf")
    got = ddbc.sql(f"select {Lit(value)} from t").fetchone()[0]
    assert got == value or (math.isnan(value) and math.isnan(got))


@pytest.mark.parametrize(
    "value,expected", [(np.int64(5), 5), (np.float64(2.5), 2.5), (np.bool_(True), True), (np.int8(-3), -3)]
)
def test_numpy_numbers_are_written_as_python_ones(ddbc, value, expected):
    assert ddbc.sql(f"select {Lit(value)}").fetchone()[0] == expected


def test_a_timezone_aware_datetime(ddbc):
    when = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    assert ddbc.sql(f"select {Lit(when)}").fetchone()[0] == when


@pytest.mark.parametrize(
    "value,error",
    [
        ("a\x00b", ValueError),
        (decimal.Decimal("NaN"), ValueError),
        ({"k": 1}, TypeError),
        (object(), TypeError),
        ({1, 2}, TypeError),
    ],
)
def test_things_which_cant_be_values_are_refused_where_they_are_made(value, error):
    with pytest.raises(error):
        Lit(value)


def test_literals_as_they_have_always_been_written():
    """Everything which used to work gives what it did before, so it can be put in its place."""
    from countess.utils.duckdb import duckdb_escape_literal  # pylint: disable=import-outside-toplevel

    for value in [
        "x",
        "it's",
        5,
        -5,
        1.5,
        -2.5e-7,
        True,
        False,
        None,
        [1, 2],
        decimal.Decimal("1.50"),
        decimal.Decimal("0.005"),
    ]:
        assert str(Lit(value)) == duckdb_escape_literal(value), value


# ---- functions and types


@pytest.mark.parametrize("name", ["sum", "SUM", "string_agg", "percent_rank", "_x1"])
def test_function_names(name):
    assert str(Fn(name)) == name


@pytest.mark.parametrize("name", ["sum(x)", "sum; drop table t", "a b", "", "1x", "x-y", None, "a.b", "sum --"])
def test_things_which_arent_function_names(name):
    with pytest.raises(ValueError):
        Fn(name)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BIGINT", "BIGINT"),
        ("DECIMAL(10, 2)", "DECIMAL(10,2)"),
        (" decimal ( 10 ,2 ) ", "decimal(10,2)"),
        ("INTEGER[]", "INTEGER[]"),
        ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP WITH TIME ZONE"),
        ("VARCHAR[3]", "VARCHAR[3]"),
        ("DOUBLE   PRECISION", "DOUBLE PRECISION"),
    ],
)
def test_type_names(ddbc, name, expected):
    assert str(Type(name)) == expected
    ddbc.sql(f"select cast(NULL as {Type(name)})")  # (and DuckDB knows what it is)


@pytest.mark.parametrize(
    "name", ["INT); drop table t; --", "STRUCT(a INTEGER)", "", "A,B", "DECIMAL(1,2,3)", None, "BIGINT)"]
)
def test_things_which_arent_type_names(name):
    with pytest.raises(ValueError):
        Type(name)


# ---- templates


def test_a_template_has_fragments_put_in_it(ddbc):
    ddbc.sql("create table t as select * from (values (1, 'x'), (2, 'y')) v(n, s)")
    query = Sql.template(
        "SELECT {cols} FROM {table} WHERE {col} > {val} ORDER BY {col}",
        cols=join([Id("s"), Id("n")]),
        table=Id("t"),
        col=Id("n"),
        val=Lit(1),
    )
    assert str(query) == 'SELECT "s", "n" FROM "t" WHERE "n" > 1 ORDER BY "n"'
    assert query.relation(ddbc).fetchall() == [("y", 2)]


def test_the_template_is_not_an_fstring_so_braces_of_its_own_are_doubled(ddbc):
    query = Sql.template("SELECT {{'a': {val}}} AS s", val=Lit(1))
    assert str(query) == "SELECT {'a': 1} AS s"
    assert ddbc.sql(str(query)).fetchone() == ({"a": 1},)


def test_a_name_which_isnt_given_is_an_error_not_braces_in_the_query():
    """(Which is what a string which should have been an f-string but isn't makes: FROM {table_name} in the query.)"""
    with pytest.raises(KeyError, match="table.*wasn't given"):
        Sql.template("SELECT * FROM {table}")


def test_a_name_which_isnt_in_the_template_is_an_error():
    with pytest.raises(TypeError, match="oops"):
        Sql.template("SELECT {a}", a=Id("a"), oops=Id("b"))


@pytest.mark.parametrize("plain", ["a", 5, None, ["a"], b"x"])
def test_a_plain_value_is_refused_so_what_kind_it_is_has_to_be_said(plain):
    with pytest.raises(TypeError, match="Id, Lit"):
        Sql.template("SELECT {x}", x=plain)
    with pytest.raises(TypeError):
        join([Id("a"), plain])


@pytest.mark.parametrize(
    "template",
    ["SELECT {a.b}", "SELECT {a[0]}", "SELECT {a!r}", "SELECT {a:>10}", "SELECT {}", "SELECT {0}", "SELECT {a b}"],
)
def test_a_template_only_has_plain_names_in_braces(template):
    with pytest.raises((ValueError, KeyError)):
        Sql.template(template, a=Id("a"))


def test_sql_can_be_put_into_sql():
    inner = Sql.template("SELECT {c} FROM {t}", c=Id("c"), t=Id("t"))
    outer = Sql.template("SELECT * FROM ({inner}) WHERE {c} > {v}", inner=inner, c=Id("c"), v=Lit(0))
    assert str(outer) == 'SELECT * FROM (SELECT "c" FROM "t") WHERE "c" > 0'


def test_a_fragment_cannot_be_added_to_text_so_it_cant_slip_out_of_its_class():
    with pytest.raises(TypeError):
        Id("a") + " and more"  # pylint: disable=expression-not-assigned


def test_join():
    assert str(join([Id("a"), Id("b")])) == '"a", "b"'
    assert str(join([Lit(1), Lit(2)], sep=" AND ")) == "1 AND 2"
    assert str(join([])) == "" and not join([]) and join([Id("a")])
    assert join([]) == EMPTY


def test_fragments_compare_by_kind_and_text():
    assert Id("a") == Id("a") and Id("a") != Raw('"a"') and Id("a") != '"a"'
    assert len({Id("a"), Id("a"), Lit("a")}) == 2
    assert repr(Id("a")) == 'Id("a")' and isinstance(Id("a"), Fragment)


# ---- text which is part of a pattern, or of a name made from one


@pytest.mark.parametrize(
    "text", AWKWARD_NAMES + ["a+b", "x[1]", "^x$", "c{2}", "a|b", "100%", "a~b", "#h", "tab\there", "back\\slash", ""]
)
def test_regex_escaped_text_matches_exactly_itself(ddbc, text):
    pattern = Lit(regex_escape(text))
    assert ddbc.sql(f"select regexp_full_match({Lit(text)}, {pattern})").fetchone() == (True,)
    assert ddbc.sql(f"select regexp_full_match({Lit(text + 'x')}, {pattern})").fetchone() == (False,)


def test_regex_escaped_text_in_random_strings_matches_exactly_itself(ddbc):
    rng = random.Random(7)
    alphabet = list("()[]{}?*+-|^$\\.&~# \t\n_'\"%/;:,=!@") + ["é", "日"] + list("ab01")
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        assert ddbc.sql(f"select regexp_full_match({Lit(text)}, {Lit(regex_escape(text))})").fetchone() == (
            True,
        ), repr(text)


@pytest.mark.parametrize("prefix", ["plain", "back\\slash", "\\1", "x\\", "a b", "q'uote", "(n)"])
def test_the_alias_of_columns_has_the_match_in_it_and_a_backslash_is_a_backslash(ddbc, prefix):
    ddbc.sql('create table p as select 1 as "v_~n"')
    alias = columns_alias(prefix + "__", suffix="!")
    assert ddbc.sql(f"select COLUMNS('^(.*)_~n$') AS {alias} from p").columns == [f"{prefix}__v!"]
