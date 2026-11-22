import numpy as np

from countess.plugins.correlation import CorrelationPlugin

X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Y = [2.0, 4.0, 5.0, 4.0, 5.0, 9.0]
Z = [9.0, 7.0, 6.0, 4.0, 3.0, 1.0]


def expected(a, b):
    """Pearson's r and the population covariance, worked out with numpy"""
    return float(np.corrcoef(a, b)[0, 1]), float(np.cov(a, b, bias=True)[0, 1])


def values_table(make_source, **columns):
    names = list(columns)
    rows = ", ".join(
        "(" + ", ".join("NULL" if v is None else repr(v) for v in row) + ")" for row in zip(*columns.values())
    )
    return make_source(f"select * from (values {rows}) t({', '.join(names)})")


def correlate(ddbc, source, set_columns, selected, group=None):
    plugin = CorrelationPlugin()
    plugin.prepare(ddbc, source)
    set_columns(plugin.columns, {c: True for c in selected})
    if group:
        plugin.group.value = group
    return plugin, plugin.execute(ddbc, source)


def test_correlate_two_columns(ddbc, make_source, set_columns, assert_rows):
    source = values_table(make_source, x=X, y=Y)
    _, output = correlate(ddbc, source, set_columns, ["x", "y"])
    assert output.columns == ["column_x", "column_y", "correlation_coefficient", "covariance_population"]
    assert_rows(output, [("x", "y", *expected(X, Y))], tolerance=1e-9)


def test_every_pair_of_the_columns_chosen(ddbc, make_source, set_columns, assert_rows):
    source = values_table(make_source, x=X, y=Y, z=Z)
    _, output = correlate(ddbc, source, set_columns, ["x", "y", "z"])
    assert_rows(
        output,
        [("x", "y", *expected(X, Y)), ("x", "z", *expected(X, Z)), ("y", "z", *expected(Y, Z))],
    )
    # (and a column is not correlated with itself, nor each pair done twice)
    _, two = correlate(ddbc, source, set_columns, ["y", "z"])
    assert_rows(two, [("y", "z", *expected(Y, Z))])


def test_perfect_and_opposite_correlation(ddbc, make_source, set_columns, assert_rows):
    source = values_table(make_source, a=[1.0, 2.0, 3.0], b=[2.0, 4.0, 6.0], c=[3.0, 2.0, 1.0])
    _, output = correlate(ddbc, source, set_columns, ["a", "b", "c"])
    assert_rows(
        output,
        [("a", "b", 1.0, 4 / 3), ("a", "c", -1.0, -2 / 3), ("b", "c", -1.0, -4 / 3)],
    )


def test_rows_missing_either_value_are_left_out(ddbc, make_source, set_columns, assert_rows):
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, None]
    y = [2.0, 4.0, 5.0, 4.0, 5.0, 9.0, None, 1.0]
    source = values_table(make_source, x=x, y=y)
    _, output = correlate(ddbc, source, set_columns, ["x", "y"])
    assert_rows(output, [("x", "y", *expected(X, Y))])  # just the six complete pairs


def test_grouped(ddbc, make_source, set_columns, assert_rows):
    xa, ya = [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 5.0]
    xb, yb = [10.0, 20.0, 30.0, 40.0], [5.0, 4.0, 2.0, 1.0]
    source = values_table(make_source, g=["a"] * 4 + ["b"] * 4, x=xa + xb, y=ya + yb)
    _, output = correlate(ddbc, source, set_columns, ["x", "y"], group="g")
    assert output.columns == ["g", "column_x", "column_y", "correlation_coefficient", "covariance_population"]
    assert_rows(
        output,
        [
            ("a", "x", "y", *expected(xa, ya)),
            ("b", "x", "y", *expected(xb, yb)),
            (None, "x", "y", *expected(xa + xb, ya + yb)),  # and the whole lot, as the total
        ],
    )


def test_needs_two_columns(ddbc, make_source, set_columns):
    source = values_table(make_source, x=X, y=Y)
    assert correlate(ddbc, source, set_columns, [])[1] is None
    assert correlate(ddbc, source, set_columns, ["x"])[1] is None


def test_only_numeric_columns_are_offered(ddbc, make_source):
    plugin = CorrelationPlugin()
    plugin.prepare(ddbc, make_source("select 1 as n, 'text' as s, 2.5 as f"))
    assert [p.label for p in plugin.columns.params] == ["n", "f"]


def test_odd_column_names(ddbc, make_source, set_columns, assert_rows):
    source = make_source("""select * from (values (1.0, 2.0), (2.0, 4.5), (3.0, 5.0)) t("a b", "quo""te")""")
    _, output = correlate(ddbc, source, set_columns, ["a b", 'quo"te'])
    assert_rows(output, [("a b", 'quo"te', *expected([1.0, 2.0, 3.0], [2.0, 4.5, 5.0]))])
