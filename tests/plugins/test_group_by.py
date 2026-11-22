import logging
import statistics

import pytest

from countess.plugins.group_by import GroupByPlugin

# group, x (which has a NULL in it), y
DATA = """select * from (values
    ('a', 1.0, 'p'), ('a', 2.0, 'p'), ('a', 3.0, 'q'), ('a', NULL, 'q'),
    ('b', 10.0, 'r'), ('b', 10.0, 'r')) t(g, x, y)"""


def group_by(ddbc, source, set_aggregates, **params):
    """`set_aggregates` maps column names to the operations to do on them, like {"g": ["index"], "x": ["sum"]}"""
    plugin = GroupByPlugin()
    plugin.prepare(ddbc, source)
    for param in plugin.columns.params:
        for operation in set_aggregates.get(param.label, []):
            param.params[operation].value = True
    for key, value in params.items():
        getattr(plugin, key).value = value
    return plugin, plugin.execute(ddbc, source)


def test_group_and_count_by_default(ddbc, make_source, assert_rows):
    _, output = group_by(ddbc, make_source(DATA), {"g": ["index"]})
    assert output.columns == ["g", "count"]  # (if there are no aggregates it counts the rows)
    assert_rows(output, [("a", 4), ("b", 2)])


def test_every_aggregate(ddbc, make_source, assert_rows):
    operations = ["count", "nunique", "min", "max", "sum", "mean", "median", "var_pop"]
    _, output = group_by(ddbc, make_source(DATA), {"g": ["index"], "x": operations})
    assert output.columns == ["g"] + [f"x__{op}" for op in operations]
    a, b = [1.0, 2.0, 3.0], [10.0, 10.0]  # (a NULL isn't counted, or added up, or averaged)
    assert_rows(
        output,
        [
            ("a", 3, 3, min(a), max(a), sum(a), statistics.mean(a), statistics.median(a), statistics.pvariance(a)),
            ("b", 2, 1, min(b), max(b), sum(b), statistics.mean(b), statistics.median(b), statistics.pvariance(b)),
        ],
    )


def test_counts_and_distinct_counts(ddbc, make_source, assert_rows):
    _, output = group_by(ddbc, make_source(DATA), {"g": ["index"], "y": ["count", "nunique"]})
    assert_rows(output, [("a", 4, 2), ("b", 2, 1)])


def test_text_columns_have_a_minimum_and_a_maximum(ddbc, make_source, assert_rows):
    _, output = group_by(ddbc, make_source(DATA), {"g": ["index"], "y": ["min", "max"]})
    assert_rows(output, [("a", "p", "q"), ("b", "r", "r")])


def test_several_columns_to_group_by(ddbc, make_source, assert_rows):
    _, output = group_by(ddbc, make_source(DATA), {"g": ["index"], "y": ["index"], "x": ["sum"]})
    assert output.columns == ["g", "y", "x__sum"]
    assert_rows(output, [("a", "p", 3.0), ("a", "q", 3.0), ("b", "r", 20.0)])


def test_several_columns_to_aggregate(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1, 10), ('a', 2, 20), ('b', 5, 50)) t(g, x, z)")
    _, output = group_by(ddbc, source, {"g": ["index"], "x": ["sum", "max"], "z": ["mean"]})
    # (the aggregates are always in the order of the checkboxes, whichever order they were ticked in)
    assert output.columns == ["g", "x__max", "x__sum", "z__mean"]
    assert_rows(output, [("a", 2, 3, 15.0), ("b", 5, 5, 50.0)])


def test_no_columns_to_group_by_gives_a_single_row(ddbc, make_source, assert_rows):
    _, output = group_by(ddbc, make_source(DATA), {"x": ["sum", "count"]})
    assert output.columns == ["x__count", "x__sum"]
    assert_rows(output, [(5, 26.0)])
    _, rows = group_by(ddbc, make_source(DATA), {})
    assert_rows(rows, [(6,)])  # (and with no aggregates either, just a count of the rows)


def test_a_missing_value_is_a_group_of_its_own(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1), (NULL, 2), (NULL, 3), ('a', 4)) t(g, x)")
    _, output = group_by(ddbc, source, {"g": ["index"], "x": ["sum"]})
    assert_rows(output, [("a", 5), (None, 5)])


def test_odd_column_names(ddbc, make_source, assert_rows):
    source = make_source("""select * from (values ('a', 1), ('a', 2)) t("gr oup", "quo""te")""")
    _, output = group_by(ddbc, source, {"gr oup": ["index"], 'quo"te': ["sum"]})
    assert output.columns == ["gr oup", 'quo"te__sum']
    assert_rows(output, [("a", 3)])


def test_summing_text_gives_nothing_but_a_warning(ddbc, make_source, caplog):
    with caplog.at_level(logging.WARNING):
        _, output = group_by(ddbc, make_source(DATA), {"g": ["index"], "y": ["sum"]})
    assert output is None
    assert caplog.text


def test_one_row_for_every_column_of_the_input(ddbc, make_source):
    plugin = GroupByPlugin()
    plugin.prepare(ddbc, make_source(DATA))
    assert [p.label for p in plugin.columns.params] == ["g", "x", "y"]
    assert list(plugin.columns.params[0].params) == [
        "index",
        "count",
        "nunique",
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "var_pop",
    ]


def test_aggregates_can_be_joined_back_on_to_the_rows(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1), ('a', 2), ('b', 5)) t(g, x)")
    _, output = group_by(ddbc, source, {"g": ["index"], "x": ["sum"]}, join=True)
    assert_rows(output, [("a", 1, 3), ("a", 2, 3), ("b", 5, 5)])


def test_totals_can_be_joined_back_on_to_the_rows(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1), ('a', 2), ('b', 5)) t(g, x)")
    _, output = group_by(ddbc, source, {"x": ["sum"]}, join=True)
    assert_rows(output, [("a", 1, 8), ("a", 2, 8), ("b", 5, 8)])
