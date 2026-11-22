import logging

import pytest

from countess.plugins.columns import ColumnsPlugin


def configure(ddbc, source, settings):
    """settings maps column name to (type, new name)"""
    plugin = ColumnsPlugin()
    plugin.prepare(ddbc, source)
    for param in plugin.columns.params:
        if param.label in settings:
            ttype, rename = settings[param.label]
            param.ttype.value = ttype
            if rename:
                param.name.value = rename
    return plugin


def test_cast_columns(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 'x', '7'), (2, 'y', '8')) t(n, s, t)")
    plugin = configure(ddbc, source, {"n": ("STRING", None), "s": ("STRING", None), "t": ("INTEGER", None)})
    rows = plugin.execute(ddbc, source).fetchall()
    assert [type(v) for v in rows[0]] == [str, str, int]
    assert_rows(rows, [("1", "x", 7), ("2", "y", 8)])


def test_a_value_which_cant_be_cast_becomes_null(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('12'), ('twelve'), ('')) t(s)")
    output = configure(ddbc, source, {"s": ("INTEGER", None)}).execute(ddbc, source)
    assert_rows(output, [(12,), (None,), (None,)])


def test_boolean_and_float(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('true', '1.5'), ('false', '2.25')) t(b, f)")
    output = configure(ddbc, source, {"b": ("BOOLEAN", None), "f": ("FLOAT", None)}).execute(ddbc, source)
    assert_rows(output, [(True, 1.5), (False, 2.25)])


def test_rename(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 2)) t(a, b)")
    output = configure(ddbc, source, {"a": ("INTEGER", "first"), "b": ("INTEGER", None)}).execute(ddbc, source)
    assert output.columns == ["first", "b"]  # (a blank name keeps the name it had)
    assert_rows(output, [(1, 2)])


def test_columns_are_removed_by_not_giving_them_a_type(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 2, 3)) t(a, b, c)")
    output = configure(ddbc, source, {"a": ("INTEGER", None), "c": ("INTEGER", None)}).execute(ddbc, source)
    assert output.columns == ["a", "c"]
    assert_rows(output, [(1, 3)])


def test_nothing_chosen_gives_nothing(ddbc, make_source):
    source = make_source("select 1 as a")
    assert ColumnsPlugin().execute(ddbc, source) is None  # not even prepared
    assert configure(ddbc, source, {}).execute(ddbc, source) is None  # prepared, but nothing wanted


def test_odd_column_names(ddbc, make_source, assert_rows):
    source = make_source("""select 1 as "a b", 2 as "quo""te" """)
    output = configure(ddbc, source, {"a b": ("INTEGER", 'new "name"'), 'quo"te': ("STRING", "ünï")}).execute(
        ddbc, source
    )
    assert output.columns == ['new "name"', "ünï"]
    assert_rows(output, [(1, "2")])


def test_a_type_which_does_not_exist_is_reported_not_raised(ddbc, make_source, caplog):
    source = make_source("select 1 as a")
    plugin = configure(ddbc, source, {"a": ("NOT_A_TYPE", None)})
    with caplog.at_level(logging.WARNING):
        assert plugin.execute(ddbc, source) is None
    assert "NOT_A_TYPE" in caplog.text


@pytest.mark.xfail(
    strict=True,
    reason='"FLOAT" is DuckDB single precision (32 bit), but means Python\'s float (a double): 0.123456789012345 '
    "comes back as 0.12345679104328156, losing more than half its digits",
)
def test_floats_keep_their_precision(ddbc, make_source):
    source = make_source("select '0.123456789012345' as x")
    output = configure(ddbc, source, {"x": ("FLOAT", None)}).execute(ddbc, source)
    assert output.fetchone()[0] == 0.123456789012345


@pytest.mark.xfail(
    strict=True,
    reason='"INTEGER" is a 32 bit integer in DuckDB, and a value which is too big for that is quietly turned into a '
    "NULL by TRY_CAST: 3000000000 is lost, though a Python int (which is what the type is meant to be) has no limit",
)
def test_integers_are_not_limited_to_32_bits(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('2147483647'), ('2147483648'), ('3000000000')) t(s)")
    output = configure(ddbc, source, {"s": ("INTEGER", None)}).execute(ddbc, source)
    assert_rows(output, [(2147483647,), (2147483648,), (3000000000,)])
