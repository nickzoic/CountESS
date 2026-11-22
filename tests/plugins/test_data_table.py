import duckdb

from countess.plugins.data_table import DataTablePlugin


def test_data_table():
    plugin = DataTablePlugin()

    plugin.set_parameter("columns.0.name", "foo")
    plugin.set_parameter("columns.0.type", "STRING")
    plugin.set_parameter("columns.1.name", "bar")
    plugin.set_parameter("columns.1.type", "INTEGER")
    plugin.set_parameter("columns.2.name", "baz")
    plugin.set_parameter("columns.2.type", "FLOAT")
    plugin.set_parameter("rows.0.foo", "a")
    plugin.set_parameter("rows.0.bar", "1")
    plugin.set_parameter("rows.0.baz", "4.5")
    plugin.set_parameter("rows.1.foo", "b")
    plugin.set_parameter("rows.1.bar", "2")
    plugin.set_parameter("rows.1.baz", "5.6")
    plugin.set_parameter("rows.2.foo", "c")
    plugin.set_parameter("rows.2.bar", "3")
    plugin.set_parameter("rows.2.baz", "6.7")

    output = plugin.execute(duckdb, None)
    assert output.fetchall() == [("a", 1, 4.5), ("b", 2, 5.6), ("c", 3, 6.7)]

    plugin.set_parameter("columns.1.type", "FLOAT")
    plugin.set_parameter("columns.2.type", "INTEGER")
    plugin.set_parameter("columns.3.name", "hello")
    plugin.set_parameter("columns.3.type", "STRING")
    plugin.fix_columns()
    plugin.set_parameter("rows.2.hello", "world")

    output = plugin.execute(duckdb, None)
    assert output.fetchall() == [("a", 1.0, 4, ""), ("b", 2.0, 5, ""), ("c", 3.0, 6, "world")]

    plugin.set_parameter("columns.3.type", "FLOAT")
    plugin.fix_columns()

    output = plugin.execute(duckdb, None)
    assert output.fetchall() == [("a", 1.0, 4, None), ("b", 2.0, 5, None), ("c", 3.0, 6, None)]

    plugin.columns.del_row(1)
    plugin.fix_columns()

    output = plugin.execute(duckdb, None)
    assert output.fetchall() == [("a", 4, None), ("b", 5, None), ("c", 6, None)]


# ---- more of the data table


import pytest


def test_columns_with_blank_names_are_given_them():
    plugin = DataTablePlugin()
    plugin.set_parameter("columns.0.name", "")
    plugin.set_parameter("columns.0.type", "STRING")
    plugin.set_parameter("columns.1.name", "")
    plugin.set_parameter("columns.1.type", "INTEGER")
    plugin.set_parameter("columns.2.name", "named")
    plugin.set_parameter("columns.2.type", "STRING")
    plugin.fix_columns()  # (this is what gives them their names)
    assert [str(c.name) for c in plugin.columns] == ["col_1", "col_2", "named"]  # (numbered by where they are)
    plugin.set_parameter("rows.0.col_1", "x")
    plugin.set_parameter("rows.0.col_2", "5")
    plugin.set_parameter("rows.0.named", "y")
    output = plugin.execute(duckdb.connect(), None)
    assert output.columns == ["col_1", "col_2", "named"]
    assert output.fetchall() == [("x", 5, "y")]


def test_columns_which_have_never_been_named_are_given_names_too():
    plugin = DataTablePlugin()
    column = plugin.columns.add_row()  # (as the + button does)
    column.type.value = "STRING"
    plugin.fix_columns()
    assert [str(c.name) for c in plugin.columns] == ["col_1"]
    plugin.rows.add_row()
    assert plugin.execute(duckdb.connect(), None).columns == ["col_1"]


def test_a_table_with_no_rows_gives_nothing():
    plugin = DataTablePlugin()
    plugin.set_parameter("columns.0.name", "a")
    plugin.set_parameter("columns.0.type", "STRING")
    assert plugin.execute(duckdb.connect(), None) is None
    assert DataTablePlugin().execute(duckdb.connect(), None) is None  # (not even any columns)


def test_all_the_types():
    plugin = DataTablePlugin()
    for number, (name, type_) in enumerate((("s", "STRING"), ("i", "INTEGER"), ("f", "FLOAT"), ("b", "BOOLEAN"))):
        plugin.set_parameter(f"columns.{number}.name", name)
        plugin.set_parameter(f"columns.{number}.type", type_)
    plugin.fix_columns()
    for key, value in (("s", "text"), ("i", "42"), ("f", "2.5"), ("b", True)):
        plugin.set_parameter(f"rows.0.{key}", value)
    output = plugin.execute(duckdb.connect(), None)
    assert output.fetchall() == [("text", 42, 2.5, True)]
    assert [type(v) for v in plugin.execute(duckdb.connect(), None).fetchone()] == [str, int, float, bool]


def test_values_which_need_quoting_and_unicode():
    plugin = DataTablePlugin()
    plugin.set_parameter("columns.0.name", 'a "quoted" name')
    plugin.set_parameter("columns.0.type", "STRING")
    plugin.fix_columns()
    plugin.set_parameter('rows.0.a "quoted" name', 'it\'s "here" — ünï')
    output = plugin.execute(duckdb.connect(), None)
    assert output.columns == ['a "quoted" name']
    assert output.fetchall() == [('it\'s "here" — ünï',)]


def test_json_without_any_rows():
    plugin = DataTablePlugin()
    plugin.set_json_data({"columns": [{"name": "a", "type": "STRING"}, {"name": "b", "type": "INTEGER"}]})
    assert [str(c.name) for c in plugin.columns] == ["a", "b"]
    assert len(plugin.rows) == 0
    assert plugin.get_json_data()["rows"] == []


def test_the_table_is_remade_each_time_it_is_asked_for():
    plugin = DataTablePlugin()
    plugin.set_parameter("columns.0.name", "a")
    plugin.set_parameter("columns.0.type", "INTEGER")
    plugin.fix_columns()
    plugin.set_parameter("rows.0.a", "1")
    assert plugin.execute(duckdb.connect(), None).fetchall() == [(1,)]
    plugin.set_parameter("rows.1.a", "2")
    assert plugin.execute(duckdb.connect(), None).fetchall() == [(1,), (2,)]
