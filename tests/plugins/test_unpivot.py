import logging

from countess.plugins.unpivot import UnpivotPlugin


def unpivot(ddbc, source, melt, set_columns, **params):
    plugin = UnpivotPlugin()
    plugin.prepare(ddbc, source)
    set_columns(plugin.columns, {c: True for c in melt})
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin, plugin.execute(ddbc, source)


def test_unpivot(ddbc, make_source, set_columns, assert_rows):
    source = make_source("select * from (values (1, 10, 20), (2, 30, 40)) t(id, a, b)")
    _, output = unpivot(ddbc, source, ["a", "b"], set_columns)
    assert output.columns == ["id", "name", "value"]  # the names it uses if not given others
    assert_rows(output, [(1, "a", 10), (1, "b", 20), (2, "a", 30), (2, "b", 40)])


def test_unpivot_with_names(ddbc, make_source, set_columns, assert_rows):
    source = make_source("select * from (values (1, 10, 20)) t(id, a, b)")
    _, output = unpivot(ddbc, source, ["a", "b"], set_columns, name_column="which", value_column="amount")
    assert output.columns == ["id", "which", "amount"]
    assert_rows(output, [(1, "a", 10), (1, "b", 20)])


def test_only_the_columns_chosen_are_unpivoted(ddbc, make_source, set_columns, assert_rows):
    source = make_source("select * from (values (1, 10, 20, 30)) t(id, a, b, c)")
    _, output = unpivot(ddbc, source, ["b"], set_columns)
    assert output.columns == ["id", "a", "c", "name", "value"]
    assert_rows(output, [(1, 10, 30, "b", 20)])


def test_nulls_are_not_unpivoted(ddbc, make_source, set_columns, assert_rows):
    source = make_source("select * from (values (1, 10, NULL), (2, NULL, 40)) t(id, a, b)")
    _, output = unpivot(ddbc, source, ["a", "b"], set_columns)
    assert_rows(output, [(1, "a", 10), (2, "b", 40)])


def test_omit_blanks(ddbc, make_source, set_columns, assert_rows):
    source = make_source("select * from (values (1, 'p', ''), (2, '', 'q')) t(id, x, y)")
    _, everything = unpivot(ddbc, source, ["x", "y"], set_columns)
    assert_rows(everything, [(1, "x", "p"), (1, "y", ""), (2, "x", ""), (2, "y", "q")])

    _, output = unpivot(ddbc, source, ["x", "y"], set_columns, omit_blanks=True)
    assert_rows(output, [(1, "x", "p"), (2, "y", "q")])


def test_columns_which_have_gone_are_ignored(ddbc, make_source, set_columns, assert_rows):
    plugin, _ = unpivot(
        ddbc, make_source("select 1 as id, 2 as a, 3 as b"), ["a", "b"], set_columns
    )  # (it's configured to unpivot b ...)
    narrower = make_source("select 1 as id, 5 as a")  # ... but now there isn't one
    assert_rows(plugin.execute(ddbc, narrower), [(1, "a", 5)])


def test_odd_column_names(ddbc, make_source, set_columns, assert_rows):
    source = make_source("""select 1 as "i d", 2 as "a b", 3 as "quo""te" """)
    _, output = unpivot(ddbc, source, ["a b", 'quo"te'], set_columns, name_column="na me")
    assert output.columns == ["i d", "na me", "value"]
    assert_rows(output, [(1, "a b", 2), (1, 'quo"te', 3)])


def test_nothing_to_unpivot_gives_nothing(ddbc, make_source, set_columns, caplog):
    source = make_source("select 1 as id, 2 as a")
    with caplog.at_level(logging.WARNING):
        _, output = unpivot(ddbc, source, [], set_columns)
    assert output is None
    assert "syntax error" in caplog.text  # (what's wrong is only reported in the log)


def test_columns_of_different_types_cannot_be_unpivoted_together(ddbc, make_source, set_columns, caplog):
    source = make_source("select 1 as id, 2 as a, 'x' as b")
    with caplog.at_level(logging.WARNING):
        _, output = unpivot(ddbc, source, ["a", "b"], set_columns)
    assert output is None
    assert "explicit cast is required" in caplog.text
